import tiktoken
import torch


class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        with open("input.txt", "r") as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        self.tokens = enc.encode(text)

        self.current_position = B * T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T

        buf = torch.tensor(
            self.tokens[
                self.current_position:
                self.current_position + B * T + 1
            ]
        )

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)

        self.current_position += B * T * self.num_processes

        if (
            self.current_position
            + B * T * self.num_processes
            + 1
            > len(self.tokens)
        ):
            self.current_position = B * T * self.process_rank

        return x, y
