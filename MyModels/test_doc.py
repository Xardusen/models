MORSE_CODE = {'1':'sad', '2': 're', '3': 'gg'}

def decodeMorse(morse_code):
    # ToDo: Accept dots, dashes and spaces, return human-readable message
    morse_code = morse_code.strip()
    out = ''
    checking = ' '
    while morse_code:
        out += MORSE_CODE[morse_code[: morse_code.index(checking)]] + checking
        morse_code = morse_code[morse_code.index(checking) + 1 :]
        if checking not in morse_code:
            out += MORSE_CODE[morse_code]
            morse_code = ''
    return out
print(decodeMorse('1 2 3'))