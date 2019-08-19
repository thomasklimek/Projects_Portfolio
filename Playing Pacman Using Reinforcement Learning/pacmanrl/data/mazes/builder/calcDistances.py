import sys
import numpy
import Queue


def bfs(adj, source):
  
  # This will store for each node (1) whether it has been visited or not, (2) the shortest distance from the source to this node, and (3) the preceding node in the shortest path
  struct = numpy.ones((len(adj), 3), dtype=numpy.int) * -1

  # Initialize 
  struct[source, 0] = 0
  struct[source, 1] = 0
  q = Queue.Queue()
  q.put(source)

  # Main BFS loop
  while not q.empty():
    u = q.get()
    for i in range(4):
      v = adj[u, i]
      if v == -1:
        continue
      elif struct[v, 0] == -1:
        struct[v, 0] = 0                        # Mark as frontier node
        struct[v, 1] = struct[u, 1] + 1         # Increment shortest path distance
        struct[v, 2] = u                        # Keep track of predecessor
        q.put(v)
    struct[u, 0] = 1  

  return struct      


def main():

  if len(sys.argv) < 2:
    print "Missing maze file"
    return


  # Read the maze file
  mazeFile = sys.argv[1]
  outFile = 'd' + mazeFile
  f = open(mazeFile, 'r')

  # Header
  header = f.readline()
  splitHeader = header.split('\t')
  nNodes = int(splitHeader[4])
  #print nNodes

  # Adjacency matrix
  adj = numpy.ones((nNodes, 4), dtype=numpy.int) * -1
  counter = 0
  for line in f:
    splitLine = line.split('\t')
    for n in range(3,7):
      adj[counter, n-3] = int(splitLine[n])
    counter += 1
  f.close()
  #print adj 

  # Calculate shortest path distances for each node and save out
  fout = open(outFile, 'w')    
  for node in range(nNodes):
    print (node+1), '/', nNodes
    struct = bfs(adj, node)
    for pairNode in range(node+1):
      fout.write(str(struct[pairNode, 1]) + '\n')
  fout.close()


if __name__=="__main__":
  main()



