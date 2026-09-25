# Contributing to crash-cayon Amazon ML Challenge 2026

Thank you for your interest in our project! This document explains how to contribute.

## 👥 Team Membership

This is a **hackathon project** with fixed team membership:
- **Team crash-canyon** (4 members)
- Active during: Sep 25-27, 2026

External contributions are welcome but limited to:
- 🐛 Bug reports
- 📖 Documentation improvements
- 🧠 Methodology feedback
- 🔍 Code review comments

## 🐛 Reporting Bugs

Found a bug? Please open a GitHub Issue with:
1. **Clear title** — one-line description
2. **Steps to reproduce** — minimal code snippet
3. **Expected vs actual behavior**
4. **Environment** — Python version, OS, library versions
5. **Logs** — paste relevant error messages

## 📖 Documentation Improvements

Spotted a typo, unclear section, or missing info?
1. Open an Issue describing the improvement, OR
2. Open a Pull Request with the fix

## 🧠 Methodology Feedback

Disagree with a design choice? Have a better idea?
1. Open a Discussion (not Issue) — this is a hackathon, so we may be locked in
2. Be respectful — we made these choices under time pressure with limited info

## 🔀 Pull Request Process

For minor fixes (typos, bug fixes):
1. Fork the repo
2. Create a branch: `git checkout -b fix/your-fix-name`
3. Make changes
4. Test: `python -m pytest code/business_entity_resolution/tests/`
5. Submit PR with clear description

For major changes (architecture, new features):
- Open a Discussion first to align with the team
- We may not accept major changes during the hackathon

## 📝 Commit Message Convention

We follow Conventional Commits:
- `feat:` new feature
- `fix:` bug fix
- `docs:` documentation only
- `style:` formatting, no code change
- `refactor:` code change that neither fixes a bug nor adds a feature
- `test:` adding tests
- `chore:` maintenance

Examples:
- `feat: add cross-encoder reranking layer`
- `fix: handle missing business_address in S2`
- `docs: update methodology with leaderboard results`

## 🧪 Testing

Before submitting a PR:
```bash
cd code/business_entity_resolution
python -m pytest tests/ -v
```

All tests must pass. The validator test (`tests/test_format.py`) is critical — it ensures our output matches the official format.

## ⚖️ Code of Conduct

By participating, you agree to:
- Be respectful and inclusive
- Focus on constructive feedback
- Accept decisions made by the team
- Not submit code that violates the Amazon ML Challenge 2026 rules

## 🏆 Acknowledgments

Contributors will be listed in:
- `README.md` (if significant contribution)
- The hackathon final presentation (if applicable)

---

**Thank you for helping us learn and improve!** 🏔️
