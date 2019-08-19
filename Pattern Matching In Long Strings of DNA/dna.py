from collections import deque
import cProfile

'''
import objgraph
pr = cProfile.Profile()
pr.enable()
'''
class Node:
	def __init__(self):
		self.data = None
		self.children = {}
		self.suffixLink = None
		self.outputLink = None
	def __repr__(self):
		string = ''
		if self.data is not None:
			string +=str(self.data) 
		return string + str(self.children)
class Trie:
	def __init__(self, words, data):
		self.root = Node()
		for i in range(len(words)):
			self.addWord(words[i], data[i], i)
		self.makeSuffixLinks()
	def addWord(self, word, data, index):
		curr = self.root
		for char in word:
			if char not in curr.children:
				curr.children[char] = Node()
			curr = curr.children[char]
		if 'END' in curr.children:
			curr.children['END'].append((index, data))
		else:
			curr.children['END'] = [(index, data)]
			#curr.children['END'].append((index, data))

	def makeSuffixLinks(self):
		root = self.root
		q = deque()
		for child in root.children:
			root.children[child].suffixLink = root
			q.append(root.children[child])
		while q:
			curr = q.popleft()
			suff = curr.suffixLink
			for c in curr.children:
				if c is 'END':
					continue
				while suff is not None:	
					if c in suff.children:
						curr.children[c].suffixLink = suff.children[c]
						break
					suff = suff.suffixLink
				else:
					curr.children[c].suffixLink = root

				output = curr.children[c].suffixLink
				if 'END' in output.children:
					curr.children[c].outputLink = output
				else:
					curr.children[c].outputLink = output.outputLink
					
				q.append(curr.children[c])

	def getValue(self, string, low, high):
		value = 0
		curr = self.root
		for c in string:
			if c not in curr.children:
				while curr is not self.root:
					curr = curr.suffixLink
					if c in curr.children:
						curr = curr.children[c]
						break
			else:
				curr = curr.children[c]

			if 'END' in curr.children:
				for val in curr.children['END']:
					if val[0] >= low and val[0] <= high:
						value += val[1]

			output = curr.outputLink
			while output is not None:
				for val in output.children['END']:
					
					if val[0] >= low and val[0] <= high:
						value += val[1]
				output = output.outputLink

		return value

	def __repr__(self):
		return str(self.root)


num_genes = int(input())
genes = list(input().split())
health = list(map(int, input().split()))

trie = Trie(genes, health)

strands = int(input())

minVal = None
maxVal = None
for i in range(strands):
	strand = input().split()
	low = int(strand[0])
	high = int(strand[1])
	healthvalue = trie.getValue(strand[2], low, high)
	if minVal is None and maxVal is None:
		minVal, maxVal = healthvalue, healthvalue
	elif healthvalue < minVal:
		minVal = healthvalue
	elif healthvalue > maxVal:
		maxVal = healthvalue
print(minVal, maxVal)

'''
objgraph.show_most_common_types() 
pr.disable()
pr.print_stats(sort='time')
'''