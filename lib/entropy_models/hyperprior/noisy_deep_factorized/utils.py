import io
import math
from typing import List, Optional


class BytesListUtils:
    @staticmethod
    def concat_bytes_list(bytes_list: List[bytes], bs_io: io.BytesIO = None,
                          head_bits_num: int = 1) -> Optional[bytes]:
        assert len(bytes_list) > 1

        bytes_len_bytes_list = []
        bytes_len_bytes_len_list = []
        for _ in bytes_list:
            bytes_len = len(_)
            bytes_len_bytes_len = math.ceil(bytes_len.bit_length() / 8)
            assert (bytes_len_bytes_len - 1).bit_length() <= head_bits_num
            bytes_len_bytes_list.append(bytes_len.to_bytes(bytes_len_bytes_len, 'little', signed=False))
            bytes_len_bytes_len_list.append(bytes_len_bytes_len)

        head_bytes = int(
            '1' + ''.join((format(_ - 1, f'0{head_bits_num}b') for _ in bytes_len_bytes_len_list)), 2
        ).to_bytes(math.ceil(len(bytes_list) / (8 // head_bits_num) + 0.125), 'little', signed=False)

        if bs_io is None:
            return_bytes = True
            bs_io = io.BytesIO()
        else:
            return_bytes = False
        bs_io.write(head_bytes)
        for _ in bytes_len_bytes_list:
            bs_io.write(_)
        for _ in bytes_list:
            bs_io.write(_)
        if return_bytes:
            ret = bs_io.getvalue()
            bs_io.close()
            return ret

    @staticmethod
    def split_bytes_list(concat_bytes: Optional[bytes], bytes_list_len: int,
                         bs_io: io.BytesIO = None,
                         head_bits_num: int = 1) -> List[bytes]:
        if bs_io is None:
            bytes_given = True
        else:
            bytes_given = False
            assert concat_bytes is None

        if bytes_given:
            bs_io = io.BytesIO(concat_bytes)
        head_bytes_len = math.ceil(bytes_list_len / (8 // head_bits_num) + 0.125)
        head_bits = f"{int.from_bytes(bs_io.read(head_bytes_len), 'little'):b}"[1:]

        bytes_len_bytes_len_list = []
        for idx in range(0, bytes_list_len * head_bits_num, head_bits_num):
            bytes_len_bytes_len_list.append(int(head_bits[idx: idx + head_bits_num], 2) + 1)

        bytes_len_list = []
        for bytes_len_bytes_len in bytes_len_bytes_len_list:
            bytes_len_list.append(int.from_bytes(bs_io.read(bytes_len_bytes_len), 'little'))

        bytes_list = []
        for bytes_len in bytes_len_list:
            bytes_list.append(bs_io.read(bytes_len))

        if bytes_given:
            bs_io.close()
        return bytes_list
