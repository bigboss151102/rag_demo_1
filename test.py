import json

# Chuỗi JSON mã hóa (ví dụ từ `page_content`)
encoded_content = "\\u0110\\u00f3ng k\\u1ef9 sau khi s\\u1eed d\\u1ee5ng..."

# Giải mã JSON
decoded_content = json.loads(f'"{encoded_content}"')
print(decoded_content)
