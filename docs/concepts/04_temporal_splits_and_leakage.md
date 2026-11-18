# Chronological splits and information leakage

Training estimates parameters. Validation selects configurations and checkpoints.
Test assesses the frozen selected system. Their roles differ even when the same
metric formula is used.

| Split | Half-open interval | Frames |
| --- | --- | ---: |
| Training | `[0,21024)` | 21,024 |
| Validation | `[21024,28032)` | 7,008 |
| Test | `[28032,35040)` | 7,008 |

A half-open interval includes its start and excludes its end. The first 60% is
training, the final 20% test, and validation receives the rounding remainder.
Time is split before constructing examples; dates are not randomly mixed.

## Leakage examples

Fitting a scaler on all dates, deriving similarity from test traffic, selecting
hidden width from test scores, or using target traffic in an input would leak
information across the intended prediction boundary.

Our scaler and seasonal table use training observations only. Every target
window stays within one split. Earlier split observations may provide history:
the first validation window uses the final training history. That history was
already available at the forecast origin.

Training-window shuffling does not reorder readings inside a window or move
examples into another split.

## Reused test limitation

The previous project tested this period, and the small LSTM was tested before
the contextual redesign. Selecting contextual candidates on validation avoids
additional direct test tuning, but does not make this test period newly unseen.
The report must disclose that chronology and the follow-up nature of the study.

See [experiment fairness](17_comparison_and_fairness.md).
