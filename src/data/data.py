"""Data modules for language-model training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import requests
import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset

from ..core.base import BaseDataModule
from ..core.registry import DATA_REGISTRY


class TokenBlockDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, token_ids: list[int] | torch.Tensor, context_length: int) -> None:
        if len(token_ids) <= context_length:
            raise ValueError(
                f"The dataset needs more than context_length={context_length} tokens; "
                f"only {len(token_ids)} were found."
            )
        if isinstance(token_ids, torch.Tensor):
            values = token_ids.to(dtype=torch.long)
        else:
            values = torch.tensor(token_ids, dtype=torch.long)
        self.inputs = values[:-1]
        self.targets = values[1:]
        self.context_length = context_length

    def __len__(self) -> int:
        return max(0, (len(self.inputs) - self.context_length) // self.context_length + 1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = index * self.context_length
        end = start + self.context_length
        return {"input_ids": self.inputs[start:end], "target_ids": self.targets[start:end]}


SUDOKU_SIZE = 9
SUDOKU_CELL_COUNT = SUDOKU_SIZE * SUDOKU_SIZE
SUDOKU_DIGIT_COUNT = SUDOKU_SIZE + 1  # 0 is the blank/clue token.
SUDOKU_POSITION_OFFSET = SUDOKU_DIGIT_COUNT
SUDOKU_SEPARATOR_TOKEN = SUDOKU_POSITION_OFFSET + SUDOKU_CELL_COUNT
SUDOKU_END_TOKEN = SUDOKU_SEPARATOR_TOKEN + 1
SUDOKU_VOCAB_SIZE = SUDOKU_END_TOKEN + 1


def encode_sudoku_prompt(prompt: str) -> list[int]:
    """Validate and encode an 81-character Sudoku grid for causal inference."""
    if len(prompt) != SUDOKU_CELL_COUNT:
        raise ValueError(f"Sudoku prompt must contain exactly {SUDOKU_CELL_COUNT} characters.")
    if any(character not in "0123456789" for character in prompt):
        raise ValueError("Sudoku prompt may contain only digits 0-9.")

    board = [int(character) for character in prompt]
    expected = set(range(1, SUDOKU_SIZE + 1))
    for unit in SUDOKU_UNITS:
        clues = [board[index] for index in unit if board[index] != 0]
        if len(clues) != len(set(clues)):
            raise ValueError("Sudoku prompt contains contradictory clues.")
        if any(value not in expected for value in clues):
            raise ValueError("Sudoku prompt contains an invalid clue.")
    return [*board, SUDOKU_SEPARATOR_TOKEN]


@dataclass(frozen=True)
class SudokuExample:
    """A generated Sudoku puzzle, solution, and serialized completion trace."""

    puzzle: tuple[int, ...]
    solution: tuple[int, ...]
    token_ids: tuple[int, ...]


class SudokuDataset(Dataset[dict[str, torch.Tensor]]):
    """Causal-LM examples for serialized Sudoku completion traces."""

    def __init__(
        self,
        examples: list[SudokuExample],
        context_length: int | None = None,
        pad_token_id: int = SUDOKU_END_TOKEN,
    ) -> None:
        if not examples:
            raise ValueError("A Sudoku dataset must contain at least one example.")
        self.examples = examples
        self.context_length = context_length
        self.pad_token_id = pad_token_id

        first_len = len(examples[0].token_ids)
        all_same_len = all(len(example.token_ids) == first_len for example in examples)

        if all_same_len:
            values = torch.tensor([example.token_ids for example in examples], dtype=torch.long)
            self.inputs = values[:, :-1]
            self.targets = values[:, 1:]
        else:
            max_seq_len = (
                context_length
                if context_length is not None
                else max(len(ex.token_ids) - 1 for ex in examples)
            )
            input_list: list[torch.Tensor] = []
            target_list: list[torch.Tensor] = []
            for example in examples:
                tokens = example.token_ids
                seq_in = tokens[:-1]
                seq_tgt = tokens[1:]
                if len(seq_in) > max_seq_len:
                    seq_in = seq_in[:max_seq_len]
                    seq_tgt = seq_tgt[:max_seq_len]
                in_tensor = torch.full((max_seq_len,), self.pad_token_id, dtype=torch.long)
                tgt_tensor = torch.full((max_seq_len,), -100, dtype=torch.long)
                in_tensor[: len(seq_in)] = torch.tensor(seq_in, dtype=torch.long)
                tgt_tensor[: len(seq_tgt)] = torch.tensor(seq_tgt, dtype=torch.long)
                input_list.append(in_tensor)
                target_list.append(tgt_tensor)
            self.inputs = torch.stack(input_list)
            self.targets = torch.stack(target_list)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"input_ids": self.inputs[index], "target_ids": self.targets[index]}


def _sudoku_units() -> list[tuple[int, ...]]:
    rows = [tuple(row * SUDOKU_SIZE + column for column in range(SUDOKU_SIZE)) for row in range(SUDOKU_SIZE)]
    columns = [tuple(row * SUDOKU_SIZE + column for row in range(SUDOKU_SIZE)) for column in range(SUDOKU_SIZE)]
    boxes = [
        tuple((box_row + row) * SUDOKU_SIZE + box_column + column for row in range(3) for column in range(3))
        for box_row in range(0, SUDOKU_SIZE, 3)
        for box_column in range(0, SUDOKU_SIZE, 3)
    ]
    return rows + columns + boxes


SUDOKU_UNITS = _sudoku_units()


def is_valid_sudoku_board(board: tuple[int, ...] | list[int]) -> bool:
    """Return whether a flattened board is a complete valid Sudoku solution."""
    if len(board) != SUDOKU_CELL_COUNT or any(value not in range(1, SUDOKU_SIZE + 1) for value in board):
        return False
    expected = set(range(1, SUDOKU_SIZE + 1))
    return all({board[index] for index in unit} == expected for unit in SUDOKU_UNITS)


def _generate_solved_board(rng: random.Random) -> tuple[int, ...]:
    board = [0] * SUDOKU_CELL_COUNT

    def fill() -> bool:
        try:
            index = board.index(0)
        except ValueError:
            return True
        row, column = divmod(index, SUDOKU_SIZE)
        used = {board[row * SUDOKU_SIZE + other] for other in range(SUDOKU_SIZE)}
        used.update(board[other * SUDOKU_SIZE + column] for other in range(SUDOKU_SIZE))
        box_row, box_column = row - row % 3, column - column % 3
        used.update(
            board[(box_row + other_row) * SUDOKU_SIZE + box_column + other_column]
            for other_row in range(3)
            for other_column in range(3)
        )
        candidates = [value for value in range(1, SUDOKU_SIZE + 1) if value not in used]
        rng.shuffle(candidates)
        for value in candidates:
            board[index] = value
            if fill():
                return True
        board[index] = 0
        return False

    if not fill():
        raise RuntimeError("Could not generate a valid Sudoku board.")
    result = tuple(board)
    if not is_valid_sudoku_board(result):
        raise RuntimeError("Generated Sudoku board failed validation.")
    return result


def _serialize_sudoku(puzzle: tuple[int, ...], solution: tuple[int, ...]) -> tuple[int, ...]:
    tokens = list(puzzle)
    tokens.append(SUDOKU_SEPARATOR_TOKEN)
    for position, value in enumerate(solution):
        if puzzle[position] == 0:
            tokens.extend((SUDOKU_POSITION_OFFSET + position, value))
    tokens.append(SUDOKU_END_TOKEN)
    return tuple(tokens)


def _sample_clue_count(
    clues_spec: int | list[int] | tuple[int, ...], rng: random.Random
) -> int:
    if isinstance(clues_spec, int):
        return clues_spec
    if isinstance(clues_spec, (list, tuple)):
        if len(clues_spec) == 2:
            return rng.randint(clues_spec[0], clues_spec[1])
        if len(clues_spec) > 2:
            return rng.choice(list(clues_spec))
        if len(clues_spec) == 1:
            return clues_spec[0]
    raise ValueError(f"Invalid clues specification: {clues_spec}")


def _validate_clues_spec(clues_spec: Any) -> None:
    if isinstance(clues_spec, int):
        if not 0 <= clues_spec <= SUDOKU_CELL_COUNT:
            raise ValueError(f"clues must be between 0 and {SUDOKU_CELL_COUNT}, got {clues_spec}.")
    elif isinstance(clues_spec, (list, tuple)):
        if not clues_spec:
            raise ValueError("clues range/list cannot be empty.")
        for c in clues_spec:
            if not isinstance(c, int) or not (0 <= c <= SUDOKU_CELL_COUNT):
                raise ValueError(f"Each clue in {clues_spec} must be an integer between 0 and {SUDOKU_CELL_COUNT}.")
        if len(clues_spec) == 2 and clues_spec[0] > clues_spec[1]:
            raise ValueError(f"clues min ({clues_spec[0]}) cannot be greater than clues max ({clues_spec[1]}).")
    else:
        raise ValueError(f"clues must be an int or list/tuple of ints, got {type(clues_spec).__name__}.")


def _get_min_clues(clues_spec: int | list[int] | tuple[int, ...]) -> int:
    if isinstance(clues_spec, int):
        return clues_spec
    if isinstance(clues_spec, (list, tuple)):
        return min(clues_spec)
    raise ValueError(f"Invalid clues specification: {clues_spec}")


def _generate_sudoku_examples(
    count: int,
    clues: int | list[int] | tuple[int, ...],
    seed: int,
    forbidden_puzzles: set[tuple[int, ...]] | None = None,
) -> list[SudokuExample]:
    rng = random.Random(seed)
    forbidden = forbidden_puzzles or set()
    examples: list[SudokuExample] = []
    while len(examples) < count:
        solution = _generate_solved_board(rng)
        curr_clues = _sample_clue_count(clues, rng)
        blank_positions = rng.sample(range(SUDOKU_CELL_COUNT), SUDOKU_CELL_COUNT - curr_clues)
        puzzle_values = list(solution)
        for position in blank_positions:
            puzzle_values[position] = 0
        puzzle = tuple(puzzle_values)
        if puzzle in forbidden or any(example.puzzle == puzzle for example in examples):
            continue
        examples.append(SudokuExample(puzzle, solution, _serialize_sudoku(puzzle, solution)))
    return examples


@DATA_REGISTRY.register("tiny_shakespeare")
class TinyShakespeareDataModule(BaseDataModule):
    """Download, tokenize, split, and batch the Tiny Shakespeare corpus."""

    def __init__(
        self,
        input_file_path: str = "data/tinyshakespeare.txt",
        data_url: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        tokenizer: str = "gpt2",
        context_length: int = 256,
        batch_size: int = 32,
        validation_fraction: float = 0.1,
        num_workers: int = 0,
        seed: int = 42,
        shuffle: bool = True,
        download_timeout: float = 30.0,
    ) -> None:
        if context_length < 1 or batch_size < 1:
            raise ValueError("context_length and batch_size must be positive.")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")
        self.input_file_path = Path(input_file_path)
        self.data_url = data_url
        self.tokenizer_name = tokenizer
        self.context_length = context_length
        self.batch_size = batch_size
        self.validation_fraction = validation_fraction
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.download_timeout = download_timeout
        try:
            self.tokenizer = _load_tokenizer(tokenizer)
        except Exception as exc:
            raise ValueError(
                f"Could not load tokenizer {tokenizer!r}. Standard tiktoken encodings may "
                "need to be downloaded once before an offline run."
            ) from exc
        self.vocab_size = self.tokenizer.n_vocab
        self.eos_token_id: int = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        self.train_dataset: TokenBlockDataset | None = None
        self.val_dataset: TokenBlockDataset | None = None

    def prepare_data(self) -> None:
        if self.input_file_path.exists():
            return
        self.input_file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = requests.get(self.data_url, timeout=self.download_timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not download training data from {self.data_url!r}.") from exc
        self.input_file_path.write_text(response.text, encoding="utf-8")

    def setup(self, stage: str | None = None) -> None:
        if not self.input_file_path.exists():
            raise FileNotFoundError(
                f"Training data was not found at {self.input_file_path}. Call prepare_data() first."
            )
        text = self.input_file_path.read_text(encoding="utf-8")
        token_ids = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})
        split_index = int(len(token_ids) * (1.0 - self.validation_fraction))
        split_index = max(self.context_length + 1, min(split_index, len(token_ids) - self.context_length - 1))
        train_ids = token_ids[:split_index]
        val_ids = token_ids[split_index:]
        self.train_dataset = TokenBlockDataset(train_ids, self.context_length)
        self.val_dataset = TokenBlockDataset(val_ids, self.context_length)
        if len(self.train_dataset) == 0 or len(self.val_dataset) == 0:
            raise ValueError("The corpus is too small for the configured context_length and split.")

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting the training DataLoader.")
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]] | None:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting the validation DataLoader.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )


def _load_tokenizer(name: str):
    if name == "byte":
        # A deterministic offline tokenizer useful for smoke tests and tiny
        # local corpora. It is still represented by a tiktoken.Encoding.
        return tiktoken.Encoding(
            name="bdh_byte",
            pat_str=r"(?s:.)",
            mergeable_ranks={bytes([value]): value for value in range(256)},
            special_tokens={"<|endoftext|>": 256},
        )
    return tiktoken.get_encoding(name)


@DATA_REGISTRY.register("sudoku")
class SudokuDataModule(BaseDataModule):
    """Generate Sudoku puzzles and deterministic cell/value completion traces."""

    def __init__(
        self,
        num_samples: int = 10_000,
        validation_fraction: float = 0.1,
        clues: int | list[int] | tuple[int, ...] = 30,
        val_clues: int | list[int] | tuple[int, ...] | None = None,
        min_clues: int | None = None,
        max_clues: int | None = None,
        batch_size: int = 32,
        context_length: int = 256,
        num_workers: int = 0,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        if num_samples < 2 or batch_size < 1 or context_length < 1:
            raise ValueError("num_samples must be at least 2; batch_size and context_length must be positive.")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between 0 and 1.")

        if min_clues is not None and max_clues is not None:
            clues = [min_clues, max_clues]
        elif min_clues is not None or max_clues is not None:
            raise ValueError("Both min_clues and max_clues must be specified if one is provided.")

        _validate_clues_spec(clues)
        if val_clues is not None:
            _validate_clues_spec(val_clues)

        self.num_samples = num_samples
        self.validation_fraction = validation_fraction
        self.clues = clues
        self.val_clues = val_clues if val_clues is not None else clues
        self.batch_size = batch_size
        self.context_length = context_length
        self.num_workers = num_workers
        self.seed = seed
        self.shuffle = shuffle
        self.vocab_size = SUDOKU_VOCAB_SIZE
        self.train_dataset: SudokuDataset | None = None
        self.val_dataset: SudokuDataset | None = None

        min_c = _get_min_clues(self.clues)
        self.sequence_length = SUDOKU_CELL_COUNT + 2 * (SUDOKU_CELL_COUNT - min_c) + 2
        if self.sequence_length - 1 > context_length:
            raise ValueError(
                f"Sudoku sequence length {self.sequence_length - 1} exceeds context_length={context_length}; "
                "increase context_length or reduce the number of blank cells."
            )

    def prepare_data(self) -> None:
        """Sudoku examples are generated locally, so no preparation is required."""

    def setup(self, stage: str | None = None) -> None:
        train_count = int(self.num_samples * (1.0 - self.validation_fraction))
        val_count = self.num_samples - train_count
        train_examples = _generate_sudoku_examples(train_count, self.clues, self.seed)
        forbidden = {example.puzzle for example in train_examples}
        val_examples = _generate_sudoku_examples(val_count, self.val_clues, self.seed + 1_000_003, forbidden)
        self.train_dataset = SudokuDataset(train_examples, context_length=self.context_length)
        self.val_dataset = SudokuDataset(val_examples, context_length=self.context_length)

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        if self.train_dataset is None:
            raise RuntimeError("Call setup() before requesting the training DataLoader.")
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]]:
        if self.val_dataset is None:
            raise RuntimeError("Call setup() before requesting the validation DataLoader.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
