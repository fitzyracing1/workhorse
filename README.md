# ax-1b/chu

Space form of the Axb predictive-echo solve.

```
ax - 1b / (c h u) = 0
x = pinv(A) @ (b / (c * h * u))
```

Divisors come only from the predicted echo. Real return corrects the hypothesis. Path is pushed before any move.

- c cabin / air. Motion inhibited at c >= 1.85
- h heat and power
- u uplink hold. No push, no move

Stack: air, eat, win, talk.

Run: `python3 ax_1b_chu_space.py`
