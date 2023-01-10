---
title: "GIT"
subtitle: ""
description: "Trying to get familiar and understand git more "
date: "2022-03-15"
tags: [git, version-control]
categories: [git, version-control]

featuredImage: ""
featuredImagePreview: ""

author: "Boda Sadallah"
authorLink: "https://twitter.com/bodasadallah"

draft: false
---

## Beatiful commands

`git log --oneline --decorate --all --graph`

`git merge --abort` ==> abort merge, and get back like it never happened

`git reset --hard ` ==> is your way to lose all uncommited work in your working directory

- git fast forward is basically that git moves the commit pointer upward to the new posotion, without creating a merge commit or anything
- you can merge with ` --no-ff` flag, to disable the fast forward merge and force git to create the merge commit

### Git Bisect

- used when something broke, and you know what did broke, but you can't figure out when did it broke
- you just give it a testing criteria to test the commit history against

## Methodology

- everything inside git is an object
- all your local branches are located in .git/refs/heads
- a branch is basically a file that appoints to a commit. a branch is bisacally a pointer to specific commit
- every commit has a parent, so to assemble branches we follow and compute their parents

## Commits

- keep added changes in commits related to the same topic
- add informative commit message
- you can add parts of changes in a single file using `-p` flag in `git add -p filename`0

## Branching

### Long-running branches

- Main branch
- Dev branch

### Short-lived branches

- features branches
- bug fixes branches

## Merging

- When the one of the two branches has the head is the same as the common ancesstor of the two branches, then we can do a fast-forward merge by putting the commits of the another branch on top the common ancesstor commit

## Rebase

- rebase puts the commits of the second brach on top of the common ancesstor commit then rebase the commits of the first branch on top of the last commit of the first branch, then it changes the history of commits

**Only use rebase to clean local commit history, don't use rebase on commits that is pushed to online**
