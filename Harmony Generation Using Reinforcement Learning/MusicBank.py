MAJOR_CHORDS = [['a','c#','e'], ['a#','d','f'], ['b', 'd#', 'f#'], ['c','e','g'], ['c#','f','g#'], ['d', 'f#', 'a'], ['d#', 'g', 'a#'], ['e', 'g#', 'b'], ['f', 'a', 'c'], ['f#', 'a#', 'c#'], ['g', 'b', 'd'], ['g#', 'c', 'd#']]
MINOR_CHORDS = [['a','c','e'], ['a#','c#','f'], ['b', 'd', 'f#'], ['c','d#','g'], ['c#','e','g#'], ['d', 'f', 'a'], ['d#', 'f#', 'a#'], ['e', 'g', 'b'], ['f', 'g#', 'c'], ['f#', 'a', 'c#'], ['g', 'a#', 'd'], ['g#', 'b', 'd#']]
DIM_CHORDS = [['a','c','d#'], ['a#','c#','e'], ['b', 'd', 'f'], ['c','d#','f#'], ['c#','e','g'], ['d', 'f', 'g#'], ['d#', 'f#', 'a'], ['e', 'g', 'a#'], ['f', 'g#', 'b'], ['f#', 'a', 'c'], ['g', 'a#', 'c#'], ['g#', 'b', 'd']]
AUG_CHORDS = [['a','c#','f'], ['a#','d','f#'], ['b', 'd#', 'g'], ['c','e','g#'], ['c#','f','a'], ['d', 'f#', 'a#'], ['d#', 'g', 'b'], ['e', 'g#', 'c'], ['f', 'a', 'c#'], ['f#', 'a#', 'd'], ['g', 'b', 'd#'], ['g#', 'c', 'e']]
ALL_NOTES_REST = ['a', 'a#', 'b', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'r']
ALL_NOTES = ['a', 'a#', 'b', 'c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#']
ALL_CHORDS = MAJOR_CHORDS.copy() + MINOR_CHORDS.copy() + DIM_CHORDS.copy() + AUG_CHORDS.copy()

def get_scale_chords(root, major=True):

	rootInt = ALL_NOTES.index(root)

	if major:
		chord_types = [0,1,1,0,0,1,2]
		interval = [0,2,4,5,7,9,11]
	else:
		chord_types = [1,2,0,1,1,0,0]
		interval = [0,2,3,5,7,8,10]

	chords = []
	for i,ct in enumerate(chord_types):
		cn = (rootInt+interval[i])%len(ALL_NOTES)
		if ct == 0:
			chords.append(MAJOR_CHORDS[cn])
		if ct == 1:
			chords.append(MINOR_CHORDS[cn])
		if ct == 2:
			chords.append(DIM_CHORDS[cn])

	return chords

def get_scale_notes(root, major=True):

	rootInt = ALL_NOTES.index(root)

	if major:
		intervals = [0,2,4,5,7,9,11]
	else:
		intervals = [0,2,3,5,7,8,10]

	notes = []
	for i in intervals:
		notes.append(get_interval(root,i))
	
	return notes

def get_interval(note, other):
	if type(other) is int:
		noteIndex = ALL_NOTES.index(note)
		intervalIndex = noteIndex + other
		return ALL_NOTES[intervalIndex%len(ALL_NOTES)]
	elif type(other) is str:
		lan = len(ALL_NOTES)
		n1 = ALL_NOTES.index(note)
		n2 = ALL_NOTES.index(other)
		pos1 = max(n2-n1, n1-n2)
		pos2 = min(n2+lan-n1, n1+lan-n2)
		return min(pos1, pos2)

def chord_in(chord, chords):
	for c in chords:
		if chord_eq(chord, c):
			return True
	return False

def chord_eq(chord1, chord2):
	if len(chord1) != len(chord2):
		return False
	for note in chord1:
		if note not in chord2:
			return False
	return True

def get_chord_name(chord):
	found = []
	for c in ALL_CHORDS:
		allThere = True
		for note in chord:
			if note not in c:
				allThere = False
		if allThere:
			found = c
			break

	if found in MAJOR_CHORDS:
		return found[0]+'major'
	if found in MINOR_CHORDS:
		return found[0]+'minor'
	if found in DIM_CHORDS:
		return found[0]+'dim'
	if found in AUG_CHORDS:
		return found[0]+'aug'
	return ''

def sort_notes(notes):
	return sorted(notes, key=ALL_NOTES.index)

def test():
	print(get_scale_chords('a'))
	print(get_scale_chords('c'))
	print(get_scale_chords('g#', False))
	print(get_scale_notes('f'))
	print(get_scale_notes('g#', False))
	print(get_interval('a', 5))
	print(get_interval('a', 12))
	print(get_interval('a', 'g#'))
	print(get_interval('f', 'd'))
	print(sort_notes(['f', 'd', 'a', 'f#']))
	print(chord_eq(['c','e','a'], ['a','c','e']))

if __name__ == '__main__':
	test()

