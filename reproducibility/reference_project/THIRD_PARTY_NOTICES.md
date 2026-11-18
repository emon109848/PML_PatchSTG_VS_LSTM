# Third-party notices

## PatchSTG reference

Code-aligned adaptations in `data/spatial.py` and `models/patchstg.py` reference
[LMissher/PatchSTG](https://github.com/LMissher/PatchSTG/tree/8b298b230227a304a082e05241c69dca3740a73d),
commit `8b298b230227a304a082e05241c69dca3740a73d`.
The original license is retained below. This project is not the official release.

MIT License

Copyright (c) 2024 LMissher

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## BigST research reference

The independent short-history model in `models/bigst.py` implements mechanisms
from Han et al., *BigST: Linear Complexity Spatio-Temporal Graph Neural Network
for Traffic Forecasting on Large-Scale Road Networks*, PVLDB 17(5), 2024,
DOI 10.14778/3641204.3641217. Author source settings were inspected at commit
`830cb529354c0cda75c1660d1a0af2ffc7e66abb` of usail-hkust/BigST.
No author repository code is vendored; no permissive repository license is
assumed. This does not confer author-repository licensing on our independent
implementation. The model omits LSFE and spatial-prior regularization and is
not an official or full-method reproduction. Differences are recorded in
[the implementation guide](docs/bigst_implementation.md).
