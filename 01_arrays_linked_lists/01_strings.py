def string_reverser(our_string):
	"""
	Reverse the input string
	Args:
		our_string(string) : String to be reversed 
	Returns:
		string : The reversed string
	"""
	string_position = len(our_string) - 1
	reversed_string = ''

	while string_position>=0:
		reversed_string += our_string[string_position]
		string_position -= 1
	return reversed_string

# TEST
print("Pass" if ('retaw' == string_reverser('water')) else "Fail")
print('Pass' if ('!noitalupinam gnirts gnicitcarP' == 
	string_reverser('Practicing string manipulation!')) else 'Fail')
print('Pass' if ('3432 :si edoc esuoh ehT' == 
	string_reverser('The house code is: 2343')) else 'Fail')

def string_reverser_1(our_string):
	"""
		Reverse the input string
	Args:
		our_string(string) : String to be reversed 
	Returns:
		string : The reversed string
	"""
	new_string = "" # New empty string

	# Iterate over old string
	for i in range(len(our_string)-1,-1,-1):
		# Copy character from back of input string to empty string
		new_string += our_string[i]

	# Return reversed string
	return new_string

# TEST
print("Pass" if ('retaw' == string_reverser_1('water')) else "Fail")
print('Pass' if ('!noitalupinam gnirts gnicitcarP' == 
	string_reverser_1('Practicing string manipulation!')) else 'Fail')
print('Pass' if ('3432 :si edoc esuoh ehT' == 
	string_reverser_1('The house code is: 2343')) else 'Fail')

def anagram_checker(str1, str2):
	"""
	Check if two string are ANAGRAMS

	Args:
		str1(string), str2(string) Strings to be checked 
	Returns:
		bool: Indicates whether strings are Anagrams
	"""
	# Convert strings to their LowerCase
	low_str1 = str1.replace(" ","").lower()
	low_str2 = str2.replace(" ","").lower()
	return True if sorted(low_str1)==sorted(low_str2) else False

print("anagram_checker")
# Test Cases
print ("Pass" if not (anagram_checker('water','waiter')) else "Fail")
print ("Pass" if anagram_checker('Dormitory','Dirty room') else "Fail")
print ("Pass" if anagram_checker('Slot machines', 'Cash lost in me') else "Fail")
print ("Pass" if not (anagram_checker('A gentleman','Elegant men')) else "Fail")
print ("Pass" if anagram_checker('Time and tide wait for no man','Notified madman into water') else "Fail")

print("Word Flipper")

def word_flipper(our_string):
	"""
	Flip the individual word in a sentence
	Args:
		our_string(string): Input string to flip words
	Reurns:
		string: String with flipped words
	"""
	our_string_separated = our_string.split(sep=' ')
	words_reversed = []
	for string in our_string_separated:
		words_reversed.append(string_reverser_1(string))

	return " ".join(words_reversed)

# Test Cases

print ("Pass" if ('retaw' == word_flipper('water')) else "Fail")
print ("Pass" if ('sihT si na elpmaxe' == word_flipper('This is an example')) else "Fail")
print ("Pass" if ('sihT si eno llams pets rof ...' == word_flipper('This is one small step for ...')) else "Fail")

def word_flipper_1(our_string):
	"""
	Flip the individual word in a sentence
	Args:
		our_string(string): Input string to flip words
	Reurns:
		string: String with flipped words
	"""
	string_separated_ls = our_string.split(" ")
	for i in range(len(string_separated_ls)):
		string_separated_ls[i] = string_separated_ls[i][::-1]
	return (" ").join(string_separated_ls)

print ("Pass" if ('retaw' == word_flipper_1('water')) else "Fail")
print ("Pass" if ('sihT si na elpmaxe' == word_flipper_1('This is an example')) else "Fail")
print ("Pass" if ('sihT si eno llams pets rof ...' == word_flipper_1('This is one small step for ...')) else "Fail")

def hamming_distance(str1, str2):
	"""
	Calculate Hamming Distance between two strings
	Args:
		str1(string), str2(string) : Input strings for finding Hamming Distance
	Returns:
		int : Hamming Distance between the strings
	"""
	if len(str1) != len(str2):
		print("Cannot calculate hamming distance between two unequal string lengths")
		return None
	else:
		hamm_dist = 0
		for i in range(len(str2)):
			if str1[i]==str2[i]:
				hamm_dist += 1
	return len(str1) - hamm_dist

print("Hamming Distance")
# Test Cases

print ("Pass" if (10 == hamming_distance('ACTTGACCGGG','GATCCGGTACA')) else "Fail")
print ("Pass" if  (1 == hamming_distance('shove','stove')) else "Fail")
print ("Pass" if  (None == hamming_distance('Slot machines', 'Cash lost in me')) else "Fail")
print ("Pass" if  (9 == hamming_distance('A gentleman','Elegant men')) else "Fail")
print ("Pass" if  (2 == hamming_distance('0101010100011101','0101010100010001')) else "Fail")

