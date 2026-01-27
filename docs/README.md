# Step 2 — Setup Your Analytics Platform

**Congratulations!** You've been selected for Step 2. This guide will help you set up the tools you'll use during your trial.

---

## What You'll Install

You're installing **AURA** (our automated staking analytics tool). This lets you:
- Monitor Bittensor subnet performance
- Track validator metrics and hyperparameters  
- Evaluate staking strategies

---

## Requirements

Before starting, make sure you have:
- **Python 3.10+** (check with `python3 --version`)
- **Git** installed
- A terminal/command prompt ready

---

## Setup Steps

### Step 1: Get Repository Access

You'll receive a repository URL via email. If you don't get it within 24-48 hours, contact your onboarding contact.

Once you have the URL, clone it:

```bash
git clone <REPO_URL> bittensor-aura-tools
cd bittensor-aura-tools
```

*(Replace `<REPO_URL>` with the actual URL from your email.)*

### Step 2: Install Dependencies

```bash
# Create virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate          # Linux / macOS
# OR on Windows:  venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

**This may take 5-10 minutes** while Bittensor packages are installed.

### Step 3: Verify It Works

Run this quick test (takes ~15 seconds):

```bash
python cli.py check platform
```

**Expected output:** You should see:
- Chain connectivity status
- Current block number
- TAO price in USD
- Subnet 0 hyperparameters (tempo, immunity_period, neurons, validators)

If you see this output without errors, **you're all set!** ✅

---

## What's Next?

1. **Confirm completion** — Reply to your onboarding email saying "Step 2 complete" and that `check platform` worked.

2. **Wait for trial instructions** — Our admin will send you Step 3 instructions with your specific tasks and timeline.

---

## Quick Reference

**Test connectivity:**
```bash
python cli.py check platform
```

**Explore subnet data (optional):**
```bash
python cli.py analyze --netuid 0 --top-n 5
```

---

## Need Help?

- **Setup issues?** Contact the person who sent you this document.
- **Repository access problems?** Reply to your onboarding email.
- **Questions about the trial?** Wait for your trial brief—it will cover everything.

---

## Checklist

- [ ] Received repository URL
- [ ] Cloned the repository
- [ ] Created and activated virtual environment
- [ ] Installed dependencies (`pip install -r requirements.txt`)
- [ ] Ran `python cli.py check platform` successfully
- [ ] Replied to email confirming Step 2 is complete

---

**Good luck with your trial!**

— AURA Staking Infrastructure Team
