// offcurve-oracle is the ground-truth for the IsEdwards25519Point predicate.
//
// It reads hex-encoded 32-byte values (one per line) from stdin and, for each,
// prints "1" if filippo.io/edwards25519's Point.SetBytes accepts it (i.e. the
// bytes decompress to an Edwards25519 point) or "0" otherwise — exactly the
// predicate go-algorand's crypto.IsEdwards25519Point uses. We differential-test
// curve25519-dalek's decompress against this. See crates/f1-core/tests.
package main

import (
	"bufio"
	"encoding/hex"
	"fmt"
	"os"

	"filippo.io/edwards25519"
)

func main() {
	const maxLine = 1 << 20
	sc := bufio.NewScanner(os.Stdin)
	sc.Buffer(make([]byte, maxLine), maxLine)
	w := bufio.NewWriter(os.Stdout)
	defer w.Flush()

	for sc.Scan() {
		line := sc.Text()
		if line == "" {
			continue
		}
		b, err := hex.DecodeString(line)
		if err != nil || len(b) != 32 {
			fmt.Fprintln(w, "E")
			continue
		}
		if _, err := new(edwards25519.Point).SetBytes(b); err == nil {
			fmt.Fprintln(w, "1")
		} else {
			fmt.Fprintln(w, "0")
		}
	}
}
