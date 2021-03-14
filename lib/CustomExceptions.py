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
						
def unexpectedType(object, ExpectedType):
	"""
	Returns a message comparing expected vs actual object types
	
	@type object:			any python object
	@param object:			The object passed by the user 
	@type ExpectedType:		type
	@param ExpectedType:	The expected type of object passed by user
	@rtype:					string
	@return:				Message comparing expected vs actual object types
	"""
	
	return " ".join(["Invalid object type.  \nExpected: ",
						str(ExpectedType) + "\nReceived: ",
						str(type(object))])	
						
def outOfTolerance(value, tolerance):
	"""
	Returns a message comparing a value to the allowable tolerance
	
	@type value:			number
	@param value:			The value of the variable in question
	@type tolerance:		number
	@param tolerance:		The allowable tolerance from zero
	@rtype:					string
	@return:				Message comparing a value to the allowable tolerance
	"""
	
	return " ".join(["Value exceeds tolerance.  \nTolerance: ",
						str(tolerance) + "\nReceived: ",
						str(value)])