---
icon: lucide/vault
---

# String

## Brief

In most interviews, Data Structure and Algorithms revolves around arrays and strings only with various algorithms inside it like `two pointer`, `sliding window`, `slow and fast pointer` etc.

## [Valid Anagram](https://leetcode.com/problems/valid-anagram/)

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        return sorted(s) == sorted(t)
```

## [Valid Parenthesis](https://leetcode.com/problems/valid-parentheses/)

```python
class Solution:
    def isValid(self, s: str) -> bool:
        brace_map = { ')' : '(' , '}' : '{' , ']' : '['}
        push_map = { '(' ,  '{' , '['}

        stack = []

        for char in s:
            if char in push_map:
                stack.append(char)
            if char in brace_map.keys():
                if stack:
                    pop_char = stack.pop()
                    if  pop_char== brace_map[char]:
                        continue
                    elif pop_char != brace_map[char]:
                        return False
                if not stack:
                    return False
            
        return not stack
```

## [Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
       

        stack = []
        letters_num = 'abcdefghijklmnopqrstuvwxyz0123456789'  # kept

        # Build stack from the forward string
        for i in range(len(s)):
            char = s[i].lower()
            if char in letters_num:
                stack.append(char)

        # Pop/compare while scanning the forward string again
        for j in range(len(s)):
            char = s[j].lower()

            if char not in letters_num:
                continue

            po = stack.pop()
            if po != char:
                return False

        return len(stack) == 0
```

## [Longest Substring Without Repeating Character](https://leetcode.com/problems/longest-substring-without-repeating-characters/)


## [Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)

## [Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)

## [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/)

## [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)

## [Encode and Decode Substrings](https://leetcode.com/problems/encode-and-decode-strings/)