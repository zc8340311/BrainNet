import sys
import os

import fileio
import gspan

if __name__ == '__main__':
	print 'Database: ', sys.argv[1]
	database = fileio.read_file(sys.argv[1])
	print 'Number Graphs Read: ', len(database)
	print 'Support: ', sys.argv[2],
	minsup = int((float(sys.argv[2])*len(database)))
	print minsup
	database, freq, trimmed, flabels = gspan.trim_infrequent_nodes(database, minsup)
	database = fileio.read_file(sys.argv[1], frequent = freq)
	print 'Trimmed ', len(trimmed), ' labels from the database'
	print flabels
	gspan.project(database, freq, minsup, flabels)


def Gspan(support):
	database = fileio.read_file(r"database.txt")
	minsup = int((float(support)*len(database)))
	database, freq, trimmed, flabels = gspan.trim_infrequent_nodes(database, minsup)
	database = fileio.read_file(r"database.txt", frequent = freq)
	gspan.project(database, freq, minsup, flabels)
