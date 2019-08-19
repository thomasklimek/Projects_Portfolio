'''
First Implementation of a suffix trie.
Build complexity = n + (n-1) + (n-2) + ... + 2 + 1
O(n**2)
'''

class suffixTrie1:
	def __init__(self, word):
		self.root = {}
		self.add_word(word)
	def add_word(self, word):
		word += '$'
		for i in range(len(word)):
			curr = self.root
			for char in word[i:]:
				if char not in curr:
					curr[char] = {}
				curr = curr[char]
	def follow_path(self, q):
		curr = self.root
		for char in q:
			if char not in curr:
				return None
			curr = curr[char]
		return curr
	def isLeaf(self, node):
		if not node:
			return True
		return False
	def hasLeaf(self, node):
		return '$' in node
	def getRoot(self, node):
		return self.root
	def getChildren(self, node):
		return node
class Node:
	def __init__(self):
		self.children = {}
		self.num = 0
class suffixTrie2:

	def __init__(self, word):
		self.root = Node()
		self.add_word(word)
	def add_word(self, word):
		word += '$'
		for i in range(len(word)):
			curr = self.root
			for char in word[i:]:
				if char not in curr.children:
					curr.children[char] = Node()
				curr.num += 1
				curr = curr.children[char]
				
	def follow_path(self, q):
		curr = self.root
		for char in q:
			if char not in curr.children:
				return None
			curr = curr.children[char]
		return curr
	def isLeaf(self, node):
		if not node:
			return True
		return False
	def hasLeaf(self, node):
		return '$' in node
	def getRoot(self, node):
		return self.root
	def getChildren(self, node):
		return node
	def leavesUnder(self, node):
		return node.num

def isSubstring(s, q):
	if s.follow_path(q) is None:
		return False
	return True
def isSuffix(s, q):
	end = s.follow_path(q)
	if end is None:
		return False
	elif s.hasLeaf(end):
		return True
	return False
def numOccurences(s, q):
	def leavesUnder(node):
		if s.isLeaf(node):
			return 1
		leafCount = 0
		for child in s.getChildren(node):
			leafCount += leavesUnder(node[child])
		return leafCount
	end = s.follow_path(q)
	if end is None:
		return 0
	return leavesUnder(end)
def numOcc(s, q):
	end = s.follow_path(q)
	if end is None:
		return 0
	return s.leavesUnder(end)


num_genes = int(input())
genes = list(input().split())
health = list(map(int, input().split()))
strands = int(input())

minVal = None
maxVal = None
for i in range(strands):
	strand = input().split()
	low = int(strand[0])
	high = int(strand[1])
	s = suffixTrie2(strand[2])
	healthvalue = 0
	for j in range(low, high+1):
		healthvalue += (numOcc(s, genes[j]) * health[j])
	if minVal is None and maxVal is None:
		minVal, maxVal = healthvalue, healthvalue
	elif healthvalue < minVal:
		minVal = healthvalue
	elif healthvalue > maxVal:
		maxVal = healthvalue
print(minVal, maxVal)

'''
s = suffixTrie1('mississippi')
print(isSubstring(s, 'issip')) # True
print(isSubstring(s, 'ippis')) # False

print(isSuffix(s, 'ppi')) # True
print(isSuffix(s, 'issi')) # False

print(numOccurences(s, 'mississippi')) # 1
print(numOccurences(s, 'ss')) # 2
print(numOccurences(s, 't')) # 4
del s
'''
