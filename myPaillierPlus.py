#!/usr/bin/env python3
# Portions copyright 2012 Google Inc. All Rights Reserved.
# This file has been modified by NICTA

# This file is part of pyphe.
#
# pyphe is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyphe is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyphe.  If not, see <http://www.gnu.org/licenses/>.

import random

try:
    from collections.abc import Mapping
except ImportError:
    Mapping = dict

from paillierPlus.myencoding import EncodedNumber
from paillierPlus.myutil import invert, powmod, getprimeover, isqrt
from math import gcd

DEFAULT_KEYSIZE = 2048

nsquare = 0
precomputed_h = []    # 预生成 h
COUNT = 1000
precision = 1    # 乘数，用于浮点数加密的精度保留

def generate_paillier_keypair(private_keyring=None, n_length=DEFAULT_KEYSIZE,precompute_h_count=COUNT):
    p = q = n = None
    n_len = 0
    while n_len != n_length:
        p = getprimeover(n_length // 2)
        q = p
        while q == p:
            q = getprimeover(n_length // 2)
        n = p * q
        n_len = n.bit_length()

    nsquare =  n * n
    for _ in range(precompute_h_count):
        while True:
            r = random.SystemRandom().randrange(1, n)
            if gcd(r, n) == 1:
                precomputed_h.append(powmod(r, n, nsquare))
                break
    public_key = PaillierPublicKey(n)
    private_key = PaillierPrivateKey(public_key, p, q)

    if private_keyring is not None:
        private_keyring.add(private_key)
    return public_key, private_key


class PaillierPublicKey(object):
    def __init__(self, n):
        self.g = n + 1
        self.n = n
        self.nsquare = n * n
        self.max_int = n // 3 - 1

    def __repr__(self):
        publicKeyHash = hex(hash(self))[2:]
        return "<PaillierPublicKey {}>".format(publicKeyHash[:10])

    def __eq__(self, other):
        return self.n == other.n

    def __hash__(self):
        return hash(self.n)
    def float_to_int(self,plaintext):
        # 通过 乘数 将浮点数转换成整数(精度表示保留几位小数)
        return int(plaintext*precision)
        

    def raw_encrypt(self, plaintext, r_value=None):
        plaintext = self.float_to_int(plaintext)
        if self.n - self.max_int <= plaintext < self.n:
            # Very large plaintext, take a sneaky shortcut using inverses
            neg_plaintext = self.n - plaintext  # = abs(plaintext - nsquare)
            neg_ciphertext = (self.n * neg_plaintext + 1) % self.nsquare
            nude_ciphertext = invert(neg_ciphertext, self.nsquare)
        else:
            # we chose g = n + 1, so that we can exploit the fact that
            # (n+1)^plaintext = n*plaintext + 1 mod n^2
            nude_ciphertext = (self.n * plaintext + 1) % self.nsquare

        # 首先检查是否有预计算的 h 值
        if precomputed_h:
            h = precomputed_h.pop()  # 使用并移除一个预计算的 h 值
        else:
            # 如果没有预计算的 h 值，则动态生成一个
            h = self.get_random_lt_n()
            h = powmod(h, self.n, self.nsquare)

        obfuscator = h

        # 返回加密结果
        return (nude_ciphertext * obfuscator) % self.nsquare

    def get_random_lt_n(self):
        return random.SystemRandom().randrange(1, self.n)

    def encrypt(self, value, precision=None, r_value=None):
        if isinstance(value, EncodedNumber):
            encoding = value
        else:
            encoding = EncodedNumber.encode(self, value, precision)

        return self.encrypt_encoded(encoding, r_value)

    def encrypt_encoded(self, encoding, r_value):
        # If r_value is None, obfuscate in a call to .obfuscate() (below)
        obfuscator = r_value or 1
        ciphertext = self.raw_encrypt(encoding.encoding, r_value=obfuscator)
        encrypted_number = EncryptedNumber(self, ciphertext, encoding.exponent)
        if r_value is None:
            encrypted_number.obfuscate()
        return encrypted_number


class PaillierPrivateKey(object):
    def __init__(self, public_key, p, q):
        if not p*q == public_key.n:
            raise ValueError('given public key does not match the given p and q.')
        if p == q:
            # check that p and q are different, otherwise we can't compute p^-1 mod q
            raise ValueError('p and q have to be different')
        self.public_key = public_key
        if q < p: #ensure that p < q.
            self.p = q
            self.q = p
        else:
            self.p = p
            self.q = q
        self.psquare = self.p * self.p

        self.qsquare = self.q * self.q
        self.p_inverse = invert(self.p, self.q)
        self.hp = self.h_function(self.p, self.psquare)
        self.hq = self.h_function(self.q, self.qsquare)

    @staticmethod
    def from_totient(public_key, totient):
        p_plus_q = public_key.n - totient + 1
        p_minus_q = isqrt(p_plus_q * p_plus_q - public_key.n * 4)
        q = (p_plus_q - p_minus_q) // 2
        p = p_plus_q - q
        if not p*q == public_key.n:
            raise ValueError('given public key and totient do not match.')
        return PaillierPrivateKey(public_key, p, q)

    def __repr__(self):
        pub_repr = repr(self.public_key)
        return "<PaillierPrivateKey for {}>".format(pub_repr)

    def decrypt(self, encrypted_number):
        encoded = self.decrypt_encoded(encrypted_number)
        return encoded.decode()/precision

    def decrypt_encoded(self, encrypted_number, Encoding=None):
        if not isinstance(encrypted_number, EncryptedNumber):
            raise TypeError('Expected encrypted_number to be an EncryptedNumber'
                            ' not: %s' % type(encrypted_number))

        if self.public_key != encrypted_number.public_key:
            raise ValueError('encrypted_number was encrypted against a '
                             'different key!')

        if Encoding is None:
            Encoding = EncodedNumber

        encoded = self.raw_decrypt(encrypted_number.ciphertext(be_secure=False))
        return Encoding(self.public_key, encoded,
                             encrypted_number.exponent)

    def raw_decrypt(self, ciphertext):

        if not isinstance(ciphertext, int):
            raise TypeError('Expected ciphertext to be an int, not: %s' %
                type(ciphertext))

        decrypt_to_p = self.l_function(powmod(ciphertext, self.p-1, self.psquare), self.p) * self.hp % self.p
        decrypt_to_q = self.l_function(powmod(ciphertext, self.q-1, self.qsquare), self.q) * self.hq % self.q
        return self.crt(decrypt_to_p, decrypt_to_q)

    def h_function(self, x, xsquare):
        return invert(self.l_function(powmod(self.public_key.g, x - 1, xsquare),x), x)

    def l_function(self, x, p):
        return (x - 1) // p

    def crt(self, mp, mq):
        u = (mq - mp) * self.p_inverse % self.q
        return mp + (u * self.p)

    def __eq__(self, other):
        return self.p == other.p and self.q == other.q

    def __hash__(self):
        return hash((self.p, self.q))


class PaillierPrivateKeyring(Mapping):
    def __init__(self, private_keys=None):
        if private_keys is None:
            private_keys = []
        public_keys = [k.public_key for k in private_keys]
        self.__keyring = dict(zip(public_keys, private_keys))

    def __getitem__(self, key):
        return self.__keyring[key]

    def __len__(self):
        return len(self.__keyring)

    def __iter__(self):
        return iter(self.__keyring)

    def __delitem__(self, public_key):
        del self.__keyring[public_key]

    def add(self, private_key):

        if not isinstance(private_key, PaillierPrivateKey):
            raise TypeError("private_key should be of type PaillierPrivateKey, "
                            "not %s" % type(private_key))
        self.__keyring[private_key.public_key] = private_key

    def decrypt(self, encrypted_number):
        relevant_private_key = self.__keyring[encrypted_number.public_key]
        return relevant_private_key.decrypt(encrypted_number)


class EncryptedNumber(object):
    def __init__(self, public_key, ciphertext, exponent=0):
        self.public_key = public_key
        self.__ciphertext = ciphertext
        self.exponent = exponent
        self.__is_obfuscated = False
        if isinstance(self.ciphertext, EncryptedNumber):
            raise TypeError('ciphertext should be an integer')
        if not isinstance(self.public_key, PaillierPublicKey):
            raise TypeError('public_key should be a PaillierPublicKey')

    def __add__(self, other):
        if isinstance(other, EncryptedNumber):
            return self._add_encrypted(other)
        elif isinstance(other, EncodedNumber):
            return self._add_encoded(other)
        else:
            return self._add_scalar(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, EncryptedNumber):
            raise NotImplementedError('Good luck with that...')

        if isinstance(other, EncodedNumber):
            encoding = other
        else:
            encoding = EncodedNumber.encode(self.public_key, other)
        product = self._raw_mul(encoding.encoding)
        exponent = self.exponent + encoding.exponent

        return EncryptedNumber(self.public_key, product, exponent)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __truediv__(self, scalar):
        return self.__mul__(1 / scalar)

    def ciphertext(self, be_secure=True):
        if be_secure and not self.__is_obfuscated:
            self.obfuscate()

        return self.__ciphertext

    def decrease_exponent_to(self, new_exp):
        if new_exp > self.exponent:
            raise ValueError('New exponent %i should be more negative than '
                             'old exponent %i' % (new_exp, self.exponent))
        multiplied = self * pow(EncodedNumber.BASE, self.exponent - new_exp)
        multiplied.exponent = new_exp
        return multiplied

    def obfuscate(self):
        r = self.public_key.get_random_lt_n()
        r_pow_n = powmod(r, self.public_key.n, self.public_key.nsquare)
        self.__ciphertext = self.__ciphertext * r_pow_n % self.public_key.nsquare
        self.__is_obfuscated = True

    def _add_scalar(self, scalar):
        encoded = EncodedNumber.encode(self.public_key, scalar,
                                       max_exponent=self.exponent)

        return self._add_encoded(encoded)

    def _add_encoded(self, encoded):
        if self.public_key != encoded.public_key:
            raise ValueError("Attempted to add numbers encoded against "
                             "different public keys!")

        # In order to add two numbers, their exponents must match.
        a, b = self, encoded
        if a.exponent > b.exponent:
            a = self.decrease_exponent_to(b.exponent)
        elif a.exponent < b.exponent:
            b = b.decrease_exponent_to(a.exponent)

        # Don't bother to salt/obfuscate in a basic operation, do it
        # just before leaving the computer.
        encrypted_scalar = a.public_key.raw_encrypt(b.encoding, 1)

        sum_ciphertext = a._raw_add(a.ciphertext(False), encrypted_scalar)
        return EncryptedNumber(a.public_key, sum_ciphertext, a.exponent)

    def _add_encrypted(self, other):
        if self.public_key != other.public_key:
            raise ValueError("Attempted to add numbers encrypted against "
                             "different public keys!")

        # In order to add two numbers, their exponents must match.
        a, b = self, other
        if a.exponent > b.exponent:
            a = self.decrease_exponent_to(b.exponent)
        elif a.exponent < b.exponent:
            b = b.decrease_exponent_to(a.exponent)

        sum_ciphertext = a._raw_add(a.ciphertext(False), b.ciphertext(False))
        return EncryptedNumber(a.public_key, sum_ciphertext, a.exponent)

    def _raw_add(self, e_a, e_b):
        return e_a * e_b % self.public_key.nsquare

    def _raw_mul(self, plaintext):
        if not isinstance(plaintext, int):
            raise TypeError('Expected ciphertext to be int, not %s' %
                type(plaintext))

        if plaintext < 0 or plaintext >= self.public_key.n:
            raise ValueError('Scalar out of bounds: %i' % plaintext)

        if self.public_key.n - self.public_key.max_int <= plaintext:
            # Very large plaintext, play a sneaky trick using inverses
            neg_c = invert(self.ciphertext(False), self.public_key.nsquare)
            neg_scalar = self.public_key.n - plaintext
            return powmod(neg_c, neg_scalar, self.public_key.nsquare)
        else:
            return powmod(self.ciphertext(False), plaintext, self.public_key.nsquare)
        
if __name__ == '__main__':
    import time
    import matplotlib.pyplot as plt
    import paillierPlus.mypaillier as paillier  # PHE库

    # 配置参数
    num_values = 93322  # Number of integers to encrypt
    test_key_lengths = [128, 256, 512, 1024, 2048]  # Key lengths to test (bits)

    # Store results
    encryption_times_phe = []
    decryption_times_phe = []
    encryption_times_paillierPlus = []
    decryption_times_paillierPlus = []

    for key_length in test_key_lengths:
        print(f"\n=== Testing Key Length: {key_length} bits ===")

        # Paillier (PHE库)
        print(f"Generating Paillier key pair with phe...")
        start_keygen_phe = time.time()
        pubkey_phe, privkey_phe = paillier.generate_paillier_keypair(n_length=key_length)
        end_keygen_phe = time.time()
        keygen_time_phe = end_keygen_phe - start_keygen_phe
        print(f"Key pair generation completed in {keygen_time_phe:.4f} seconds.")

        values = list(range(1, num_values + 1))

        # PHE 加密
        print("Starting encryption with PHE...")
        start_time_enc_phe = time.time()
        encrypted_values_phe = [pubkey_phe.encrypt(val) for val in values]
        end_time_enc_phe = time.time()
        total_encryption_time_phe = end_time_enc_phe - start_time_enc_phe
        encryption_times_phe.append(total_encryption_time_phe)

        # PHE 解密
        print("Starting decryption with PHE...")
        start_time_dec_phe = time.time()
        decrypted_values_phe = [privkey_phe.decrypt(enc) for enc in encrypted_values_phe]
        end_time_dec_phe = time.time()
        total_decryption_time_phe = end_time_dec_phe - start_time_dec_phe
        decryption_times_phe.append(total_decryption_time_phe)

        print(f"Encryption completed in {total_encryption_time_phe:.4f} seconds")
        print(f"Decryption completed in {total_decryption_time_phe:.4f} seconds")

        # PaillierPlus 加密
        print(f"Generating PaillierPlus key pair...")
        start_keygen_plus = time.time()
        pubkey_plus, privkey_plus = generate_paillier_keypair(n_length=key_length, precompute_h_count=num_values)
        end_keygen_plus = time.time()
        keygen_time_plus = end_keygen_plus - start_keygen_plus
        print(f"Key pair generation completed in {keygen_time_plus:.4f} seconds.")

        # PaillierPlus 加密
        print("Starting encryption with PaillierPlus...")
        start_time_enc_plus = time.time()
        encrypted_values_plus = [pubkey_plus.encrypt(val) for val in values]
        end_time_enc_plus = time.time()
        total_encryption_time_plus = end_time_enc_plus - start_time_enc_plus
        encryption_times_paillierPlus.append(total_encryption_time_plus)

        # PaillierPlus 解密
        print("Starting decryption with PaillierPlus...")
        start_time_dec_plus = time.time()
        decrypted_values_plus = [privkey_plus.decrypt(enc) for enc in encrypted_values_plus]
        end_time_dec_plus = time.time()
        total_decryption_time_plus = end_time_dec_plus - start_time_dec_plus
        decryption_times_paillierPlus.append(total_decryption_time_plus)

        print(f"Encryption completed in {total_encryption_time_plus:.4f} seconds")
        print(f"Decryption completed in {total_decryption_time_plus:.4f} seconds")

    # 绘制加密时间对比图
    plt.figure(figsize=(10, 6))
    plt.plot(test_key_lengths, encryption_times_phe, marker='o', linestyle='-', label="Paillier Encryption Time")
    plt.plot(test_key_lengths, encryption_times_paillierPlus, marker='x', linestyle='--', label="PaillierPlus Encryption Time")
    plt.title('Encryption Time vs Key Length')
    plt.xlabel('Key Length (bits)')
    plt.ylabel('Encryption Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.xticks(test_key_lengths)
    plt.savefig('encryption_time_comparison.png', bbox_inches='tight')  # Save the plot

    # 绘制解密时间对比图
    plt.figure(figsize=(10, 6))
    plt.plot(test_key_lengths, decryption_times_phe, marker='o', linestyle='-', label="Paillier Decryption Time")
    plt.plot(test_key_lengths, decryption_times_paillierPlus, marker='x', linestyle='--', label="PaillierPlus Decryption Time")
    plt.title('Decryption Time vs Key Length')
    plt.xlabel('Key Length (bits)')
    plt.ylabel('Decryption Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.xticks(test_key_lengths)
    plt.savefig('decryption_time_comparison.png', bbox_inches='tight')  # Save the plot




