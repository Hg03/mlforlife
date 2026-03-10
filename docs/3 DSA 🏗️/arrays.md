---
icon: lucide/vault
---

# 1. Array

## Brief

In most interviews, Data Structure and Algorithms revolves around array only with various algorithms inside it like `two pointer`, `sliding window`, `slow and fast pointer` etc.

## [Find a Pair That Sums to a Target](https://leetcode.com/problems/two-sum/)

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
		    # When sorting is not allowed
        hashMap = {}
        for i in range(len(nums)):
            complement = target - nums[i]
            if complement in hashMap:
                return [hashMap[complement], i]
            else:
                hashMap[nums[i]] = i
        return [-1, -1]
    
    def twoSum_sorted(self, nums: List[int], target: int) -> List[int]:
		    # When array is sorted
        left_pointer = 0
		    right_pointer = len(numbers) - 1
		    while left_pointer < right_pointer:
		        current_sum = numbers[left_pointer] + numbers[right_pointer]
		        if current_sum == target_sum:
		            return True
		        if current_sum < target_sum:
		            left_pointer += 1
		        else:
		            right_pointer -= 1

		    return False
```

## [Best Time to Buy and Sell Stocks](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
		    # Brute Force but TLE exceeded
        mx = 0
        for i in range(len(prices)):
            for j in range(i+1,len(prices)):
                if prices[i] < prices[j]:
                    mx = max(mx,prices[j] - prices[i])
        return mx
    def maxProfit_Optimized(self, prices: List[int]) -> int:
        # Kadane's Algo
        # Track the minimum at each position and calculate the maximum profit by subtracting current price with min_price.     min_price = inf
        maxProfit = 0
        for i in range(len(prices)):
            min_price = min(min_price, prices[i])
            maxProfit = max(maxProfit, prices[i] - min_price)
        return maxProfit
```

## [Contains Duplicate](https://leetcode.com/problems/contains-duplicate/)

```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        hashSet = set()
        for num in nums:
            if num in hashSet:
                return True
            else:
                hashSet.add(num)
        return False
```

## [Product of Array Except Self](https://leetcode.com/problems/product-of-array-except-self/)

```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
		    # Brute Force
        n = len(nums)
        result = [1]*n

        for i in range(n):
            product = 1
            for j in range(n):
                if i == j:
                    continue
                elif nums[j] != 0:
                    product *= nums[j]
                else:
                    product = 0
                    break
            result[i] = product
        return result
        
    def productExceptSelf_Optimized(self, nums: List[int]) -> List[int]:
        
        n = len(nums)
        prefix = [1] * n
        postfix = [1] * n
        res = [1] * n

        for i in range(1, n):
            # calculate prefix of i = (prefix of i - 1) * (nums[i - 1])
            prefix[i] = prefix[i - 1] * nums[i - 1]

            # Postfix
            # postfix starts at 1 = the postfix of len(n) - 1
        # start reverse iteration at: len(n) - 2
        # tc: iterate list O(n)
        for i in range(n-2, -1, -1):

            # calculate postfix of i = (postfix of i + 1) * (nums[i + 1])
            postfix[i] = postfix[i + 1] * nums[i + 1]

        # Result
        # tc: iterate list O(n)
        for i in range(n):

            # calculate product except self for i = (prefix of i) * (postfix of i)
            res[i] = prefix[i] * postfix[i]

        return res
        
     def productExceptSelf_Optimized2(self, nums: List[int]) -> List[int]:
        n = len(nums)
        res = [1] * n
        prefix = 1
        for i in range(n):
            res[i] *= prefix
            prefix *= nums[i]
        postfix = 1
        for i in range(n-1, -1, -1):
            res[i] *= postfix
            postfix *= nums[i]
        return res
```

## [Maximum Sum Subarray](https://leetcode.com/problems/maximum-subarray/)

```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
		    # Brute force is to loop through each subarray and calculate sum and then it's max.
        # max_sum = -inf
        # if len(nums) == 1:
        #    return nums[0]
        # for i in range(len(nums)):
        #    curr_sum = 0
        #    for j in range(i, len(nums)):
        #        curr_sum += nums[j]
        #        max_sum = max(max_sum, curr_sum)
        # return max_sum
        # Below is the optimized approach, Calculate the sum, if it goes negative, it means, we are getting some negative number which is not helping us, so update the cur_sum to 0 again and look for next elements.
        max_sum = -inf
        cur_sum = 0
        for i in range(len(nums)):
            if cur_sum < 0:
                cur_sum = 0
            cur_sum += nums[i]
            max_sum = max(max_sum, cur_sum)
        return max_sum
```

## [Container with Most Water](https://leetcode.com/problems/container-with-most-water/)

```python
def maxArea(self, height: List[int]) -> int:
    left = 0
    right = len(height) - 1
    maxArea = 0
    while left < right :
        area = (right - left) * min(height[left], height[right])
        maxArea = max(maxArea, area)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return maxArea
```

## [Rotate Array](https://leetcode.com/problems/rotate-array/)

```python
def rotate(self, nums: List[int], k: int) -> None:
    n = len(nums)
    k = k % n
    rotated = [0] * n

    for i in range(n):
        rotated[(i + k) % n] = nums[i]
    
    for i in range(n):
        nums[i] = rotated[i]
       
# For example, If k = 9, rotated position should be
# (current position + remainder) % length of input array

# [1,2,3,4,5,6,7] -> Input
# ↓
# [6,7,1,2,3,4,5] -> Output
# For 1(= index 0), (0 + 2) % 7 = 2
# For 2(= index 1), (1 + 2) % 7 = 3
# For 6(= index 5), (5 + 2) % 7 = 0
# For 7(= index 6), (6 + 2) % 7 = 1
```

## [Search in a Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array/description/)

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    # Left Sorted Array
                    right = mid - 1
                else:
                    left = mid + 1 # move to right sorted array, if didn't find target at left
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
            
        return -1
```

## [Minimum in a Rotated Sorted Array](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/)

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        # In rotated sorted array, we have that pattern like, we'll be having maximum element on left sorted array last element and minimum is at right sorted array first element.
        res = nums[0]
        l, r = 0, len(nums) - 1

        while l <= r:
            if nums[l] < nums[r]:
                res = min(nums[l], res)
            m = (l + r) // 2
            res = min(res, nums[m])
            if nums[m] >= nums[l]: # we are in left sorted array, move to right, there we have minimum
                l = m + 1
            else:
                r = m - 1
        return res

```