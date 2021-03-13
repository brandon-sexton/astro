def unexpectedArrayDim(array, expectedDim):
	"""
	Returns a message comparing expected vs actual array dimensions
	
	@type array:		numpy array
	@param array:		The array passed by the user 
	@type expectedDim:	tuple
	@param expectedDim:	The expected shape of the input numpy array
	@rtype:				string
	@return:			Message comparing expected vs actual array dimensions
	"""
	
	return " ".join(["Invalid array dimensions.  \nExpected: ",
						str(expectedDim) + "\nReceived: ",
						str(array.shape)])
		