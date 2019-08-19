import numpy

class Maze:

  UP = 3
  RIGHT = 4
  DOWN = 5
  LEFT = 6


  def __init__(self, initX=4, initY=4, mazeFile = None):
    
    # Load an existing maze if given
    if mazeFile != None:
      pass
    # Otherwise create a start node  
    else:
      self.maze = numpy.ones((1,9), dtype=numpy.int) * -1
      self.maze[0,0] = 0        # Node no.
      self.maze[0,1] = initX    # X
      self.maze[0,2] = initY    # Y
      self.maze[0,7] = -1        # Add pill?          
      print 'Initialized maze with node 0'      
      # Keep track of pill and power pill counts 
      self.pills = 0
      self.power = 0

      self.grid = numpy.ones((1000, 1000), dtype=numpy.int) * -1
      self.grid[initX, initY] = 0

  def addCol(self, startNode, length, pills=True):
    
    if length <= 0:
      return
    
    x = self.maze[startNode, 1]
    y = self.maze[startNode, 2] + 1
    
    # Initial links
    firstNode = len(self.maze)
    self.maze[startNode, self.DOWN] = len(self.maze)
    currentNode = self.addNode(x, y, startNode, -1, -1, -1, -1, -1)

    # Subsequent links
    for n in range(length-1):
      y += 1
      nextNode = self.addNode(x, y, currentNode, -1, -1, -1, -1, -1)
      self.maze[currentNode, self.DOWN] = nextNode
      currentNode = nextNode
    lastNode = len(self.maze)-1

    print 'Added nodes', firstNode, self.maze[firstNode,1:3], 'to', lastNode, self.maze[lastNode,1:3]  # Technically if there was a merge at the end this won't consider it...  

  def addRow(self, startNode, length, pills=True):
    
    if length <= 0:
      return

    x = self.maze[startNode, 1] + 1 
    y = self.maze[startNode, 2]
    
    # Initial links
    firstNode = len(self.maze)
    self.maze[startNode, self.RIGHT] = len(self.maze) 
    currentNode = self.addNode(x, y, -1, -1, -1, startNode, -1, -1)  

    # Subsequent links 
    for n in range(length-1):
      x += 1
      nextNode = self.addNode(x, y, -1, -1, -1, currentNode, -1, -1)
      self.maze[currentNode,self.RIGHT] = nextNode
      currentNode = nextNode
    lastNode = len(self.maze)-1

    print 'Added nodes', firstNode, self.maze[firstNode,1:3], 'to', lastNode, self.maze[lastNode,1:3]    # Technically if there was a merge at the end, this won't consider it...

  def addNode(self, x, y, up=-1, right=-1, down=-1, left=-1, pill=-1, power=-1):

    # Check if this x,y location already has a node
    if self.grid[x,y] != -1:
      # If it does, merge the new info with what is already there
      index = self.grid[x,y]
      self.maze[index, 3] = Maze.mergeVals(self.maze[index,3], up)
      self.maze[index, 4] = Maze.mergeVals(self.maze[index,4], right)
      self.maze[index, 5] = Maze.mergeVals(self.maze[index,5], down)
      self.maze[index, 6] = Maze.mergeVals(self.maze[index,6], left)
      self.maze[index, 7] = Maze.mergeVals(self.maze[index,7], pill)
      self.maze[index, 8] = Maze.mergeVals(self.maze[index,8], power)
      print 'Merge at node', index
      return index
    else:
      # Otherwise create new
      node = numpy.ones(9, dtype=numpy.int) * -1
      node[0] = len(self.maze)
      node[1] = x
      node[2] = y
      node[3] = up
      node[4] = right
      node[5] = down
      node[6] = left
      node[7] = pill
      node[8] = power
      self.grid[x,y] = node[0]  
      self.maze = numpy.vstack((self.maze, node))
      return node[0]

  @staticmethod
  def mergeVals(a, b):
    if a == -1:
      return b
    else:
      return a

  @staticmethod
  def opposite(direction):
    if direction == 5 or direction == 6:
      return direction-2
    elif direction == 3 or direction == 4:
      return direction+2
    else:
      return -1

  # Link for wrap arounds  
  def linkRow(self, y):
    cols = self.maze[self.maze[:,2] == y,1]
    maxCol = cols.max()
    minCol = cols.min()
    firstColNode = self.maze[numpy.logical_and(self.maze[:,2] == y, self.maze[:,1] == minCol),0]
    lastColNode = self.maze[numpy.logical_and(self.maze[:,2] == y, self.maze[:,1] == maxCol),0]
    self.linkNodes(firstColNode, lastColNode, self.LEFT) 
    print 'Linked node', firstColNode, 'to', lastColNode, 'on row', y 

  def linkCol(self, x):
    rows = self.maze[self.maze[:,1] == x, 2]
    minRow = rows.min()
    maxRow = rows.max()
    firstRowNode = self.maze[numpy.logical_and(self.maze[:,1] == x, self.maze[:,2] == minRow),0]
    lastRowNode = self.maze[numpy.logical_and(self.maze[:,1] == x, self.maze[:,2] == maxRow),0]
    self.linkNodes(firstRowNode, lastRowNode, self.UP)
    print 'Linked node', firstRowNode, 'to', lastRowNode, 'on column', x

  def linkNodes(self, node1, node2, node1Direction):
    # Node 2 will be linked to node 1 in the reciprocal direction
    self.maze[node1, node1Direction] = node2
    self.maze[node2, Maze.opposite(node1Direction)] = node1

  def getNeighbors(self, node):
    neighbors = self.maze[node, 3:7]
    return neighbors[neighbors != -1]

  def checkMaze(self):  

    success = True

    # Ideally should check that every neighbor of a node says the node is also its neighbor


    # Check 1: Every node has at least 2 neighbors (except lair)
    for n in range(len(self.maze)):
      nNeighbors = len(self.getNeighbors(n))
      if self.lair != None and self.lair == n:
        continue
      elif nNeighbors < 2:
        print "BAD NODE:", n, "\t NOT ENOUGH NEIGHBORS" 
        success = False

    if success:
      print "Maze check successful!"
    else:
      print "Maze check unsuccessful. See errors above."

    return success


  def setGhostLair(self, x, y):
    self.lairX = x
    self.lairY = y
    self.lair = len(self.maze)
    self.addNode(x, y, -1, -1, -1, -1, -1, -1)

  def setPower(self, node):
    self.maze[node, 8] = self.power
    self.power += 1

  def dispersePills(self, frequency):
    '''
      Adds pills every FREQUENCY nodes. 
    '''
    for n in range(len(self.maze)):
      if n % frequency == 0:
        self.maze[n, 7] = self.pills
        self.pills += 1

  def setStartNodes(self, pacmanStart, ghostStart):
    self.pacmanStart = pacmanStart
    self.ghostStart = ghostStart

  def findJunctions(self):
    return numpy.sum(numpy.sum(self.maze[:,3:7] == -1, axis=1) < 2)


  def save(self, mazeName):
    fout = open(mazeName + '.txt', 'w')

    # Write header
    fout.write(mazeName + '\t' + str(self.pacmanStart) + '\t' + str(self.lair) + '\t' + str(self.ghostStart) + '\t' + str(len(self.maze)) + '\t' + str(self.pills) + '\t' + str(self.power) + '\t' + str(self.findJunctions()) + '\n')
    
    # Write maze
    for n in range(len(self.maze)):
      for v in range(len(self.maze[n])):
        if (v != 0):
          fout.write('\t')
        fout.write(str(self.maze[n,v]))
      fout.write('\n')
    fout.close()
    print 'Saved maze to ', mazeName, '.txt'

def main():
  pass


if __name__=="__main__":
  main()


