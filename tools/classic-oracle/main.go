// classic-oracle is the ground truth for classic Algorand (Ed25519) account
// derivation, independent of the Rust oracle.
//
// It reads hex-encoded 32-byte seeds (one per line) from stdin and, for each,
// prints "<pubkey hex> <address>" where the key pair is Go's crypto/ed25519
// NewKeyFromSeed (RFC 8032 — the same derivation as libsodium's
// crypto_sign_ed25519_seed_keypair, which go-algorand uses) and the address
// is basics.Address.String(): base32(pk || last4(SHA512_256(pk))), no padding.
// See crates/pq-core/tests/classic_differential.rs.
package main

import (
	"bufio"
	"crypto/ed25519"
	"crypto/sha512"
	"encoding/base32"
	"encoding/hex"
	"fmt"
	"os"
)

func main() {
	sc := bufio.NewScanner(os.Stdin)
	w := bufio.NewWriter(os.Stdout)
	defer w.Flush()
	enc := base32.StdEncoding.WithPadding(base32.NoPadding)

	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			continue
		}
		seed, err := hex.DecodeString(line)
		if err != nil || len(seed) != ed25519.SeedSize {
			fmt.Fprintln(w, "E")
			continue
		}
		pk := ed25519.NewKeyFromSeed(seed).Public().(ed25519.PublicKey)
		sum := sha512.Sum512_256(pk)
		fmt.Fprintf(w, "%s %s\n", hex.EncodeToString(pk), enc.EncodeToString(append(pk, sum[28:]...)))
	}
}
