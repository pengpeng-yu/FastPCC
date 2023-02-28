import io
import math
from typing import List


class BytesListUtils:
    @staticmethod
    def concat_bytes_list(bytes_list: List[bytes]) -> bytes:
        assert len(bytes_list) >= 1
        if len(bytes_list) == 1:
            return bytes_list[0]

        bytes_len_list = []
        for _ in bytes_list:
            bytes_len = len(_)
            assert bytes_len >= 1
            bytes_len_list.append(bytes_len - 1)

        bytes_len_bytes_list = []
        bytes_len_bytes_len_list = []
        for bytes_len in bytes_len_list:
            bytes_len_bytes_len = math.ceil(bytes_len.bit_length() / 8)
            bytes_len_bytes_list.append(bytes_len.to_bytes(bytes_len_bytes_len, 'little'))
            bytes_len_bytes_len_list.append(bytes_len_bytes_len)

        bytes_len_bytes_len_bits_list = []
        for bytes_len_bytes_len in bytes_len_bytes_len_list:
            assert 0 <= bytes_len_bytes_len <= 3
            bytes_len_bytes_len_bits = f'{bytes_len_bytes_len:b}'
            if len(bytes_len_bytes_len_bits) == 1:
                bytes_len_bytes_len_bits = '0' + bytes_len_bytes_len_bits
            bytes_len_bytes_len_bits_list.append(bytes_len_bytes_len_bits)
        head_bytes = int(
            '1' + ''.join(bytes_len_bytes_len_bits_list), 2
        ).to_bytes(math.ceil(len(bytes_list) / 4 + 0.125), 'little')

        concat_bytes = head_bytes + b''.join(bytes_len_bytes_list) + b''.join(bytes_list)
        return concat_bytes

    @staticmethod
    def split_bytes_list(concat_bytes: bytes, bytes_list_len: int) -> List[bytes]:
        if bytes_list_len == 1:
            return [concat_bytes]

        bs = io.BytesIO(concat_bytes)
        head_bytes_len = math.ceil(bytes_list_len / 4 + 0.125)
        head_bits = f"{int.from_bytes(bs.read(head_bytes_len), 'little'):b}"

        bytes_len_bytes_len_list = []
        for idx in range(1, bytes_list_len * 2 + 1, 2):
            bytes_len_bytes_len_list.append(int(head_bits[idx: idx + 2], 2))

        bytes_len_list = []
        for bytes_len_bytes_len in bytes_len_bytes_len_list:
            bytes_len_list.append(int.from_bytes(bs.read(bytes_len_bytes_len), 'little') + 1)

        bytes_list = []
        for bytes_len in bytes_len_list:
            bytes_list.append(bs.read(bytes_len))

        assert bs.read() == b''
        bs.close()
        return bytes_list
