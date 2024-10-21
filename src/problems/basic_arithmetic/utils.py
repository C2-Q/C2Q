def decimal_to_binary_list(num, n_bits):
    return [int(x) for x in bin(num)[2:].zfill(n_bits)][::-1]


def decimal_to_complement_binary_list(num, n_bits):
    if num >= 0:
        # Positive number, convert to binary and pad to n_bits
        bin_str = bin(num)[2:]  # Remove '0b' prefix
        if len(bin_str) > n_bits:
            raise ValueError("Number does not fit in the specified number of bits")
        bin_str = bin_str.zfill(n_bits)  # Pad with leading zeros
    else:
        # Negative number, calculate two's complement
        num = (1 << n_bits) + num  # Equivalent to (2^n_bits) + num
        bin_str = bin(num)[2:]  # Remove '0b' prefix

        # Convert binary string to list of integers (0s and 1s)
    bin_list = [int(bit) for bit in bin_str]

    # Reverse the list for little-endian representation
    little_endian_list = bin_list[::-1]
    return little_endian_list


def complement_binary_list_to_decimal(bin_list):
    bin_list = bin_list[::-1]

    is_negative = bin_list[0] == 1

    bin_str = ''.join(str(bit) for bit in bin_list)

    if is_negative:
        inverted_bin_list = [1 - bit for bit in bin_list]
        inverted_bin_str = ''.join(str(bit) for bit in inverted_bin_list)
        decimal_value = int(inverted_bin_str, 2) + 1
        decimal_value = -decimal_value
    else:
        decimal_value = int(bin_str, 2)

    return decimal_value
