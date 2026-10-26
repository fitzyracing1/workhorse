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

## Plans

This project is free, and its license stays as is. Support it through [GitHub Sponsors](https://github.com/sponsors/fitzyracing1):

- **Free**: use the code, community support.
- **Pro ($49/month)**: priority issues and support.
- **Team ($199/month)**: priority support for your team plus early access.
