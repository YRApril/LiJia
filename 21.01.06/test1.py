res = []
with open(r'30.v30', 'rb') as fp:
    data = fp.read(1)      #type(data) === bytes
    while data:
        text = int.from_bytes(data, byteorder='big', signed=True)
        print(text)
        res.append(text)
        data = fp.read(1)

print(res)
