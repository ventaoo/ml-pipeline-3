import os
from pathlib import Path
from dotenv import load_dotenv
from ansible.parsing.vault import VaultLib, VaultSecret
from ansible.constants import DEFAULT_VAULT_ID_MATCH

def decrypt_with_ansible_lib(encrypted_file, password, output_file):
    # 读取密钥
    secret = VaultSecret(password.encode())
    vault = VaultLib([(DEFAULT_VAULT_ID_MATCH, secret)])

    # 解密文件内容
    encrypted_data = Path(encrypted_file).read_bytes()
    decrypted_data = vault.decrypt(encrypted_data).decode()

    Path(output_file).write_text(decrypted_data)
    print(f"Decryption complete. Output written to {output_file}")
    
    load_dotenv(output_file, override=True)
    print(f"Load env {output_file}...")
    os.remove(output_file)
    print(f'Delete {output_file}...')
