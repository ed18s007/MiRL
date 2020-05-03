class Node(object):
	def __init__(self, value):
		self.value = value
		self.next = None 

head = Node(2)
print("head value is ",head.value)
print("head next is ",head.next)

head.next = Node(1)
print("second Node value is ", head.next.value)
print("second node next is ", head.next.next)

head.next.next = Node(4)
head.next.next.next = Node(3)
head.next.next.next.next = Node(5)

print(head.value)
print(head.next.value)
print(head.next.next.value)
print(head.next.next.next.value)
print(head.next.next.next.next.value)
print(head.next.next.next.next.next)

print("TRAVERSING")
# Traversing a Linked List
def traverse_linkedlist(node):
	current_node = node
	while current_node is not None:
		print(current_node.value)
		current_node = current_node.next

traverse_linkedlist(head)

print("create_linked_list")

def create_linked_list(input_list):
	"""
	Function to create linked list
	@param input_list (list) : A list of integers
	@return: head node of the linked list
	"""
	try:
		head = Node(input_list.pop(0))

		while len(input_list) > 0:
			current_node = head
			while current_node.next:
				current_node = current_node.next
			current_node.next = Node(input_list.pop(0))
	except IndexError:
		head = None

	return head

def test_function(input_list, head):
	try:
		if len(input_list) == 0:
			if head is not None:
				print("Fail")  
				return
		for elem in input_list:
			if head.value != elem :
				print("Fail")
				return
			else:
				head = head.next
		print("Pass")
	except Exception as e:
		print("Fail: " + e)

input_list = [1, 2, 3, 4, 5, 6]
head = create_linked_list(input_list)
test_function(input_list, head)

input_list = [1]
head = create_linked_list(input_list)
test_function(input_list, head)

input_list = []
head = create_linked_list(input_list)
test_function(input_list, head)

# while loop
def create_linked_list_better(input_list):
	"""
	Function to create a better linked list
	@params input_list(list): a list of integers
	return: head node of the linked list
	"""
	try:
		head = Node(input_list.pop(0))
		tail = head
		while len(input_list)>0:
			tail.next = Node(input_list.pop(0))
			tail = tail.next
	except IndexError:
		head = None
	return head

print("better linked list")
input_list = [1, 2, 3, 4, 5, 6]
head = create_linked_list_better(input_list)
test_function(input_list, head)
traverse_linkedlist(head)


input_list = [1]
head = create_linked_list_better(input_list)
test_function(input_list, head)
traverse_linkedlist(head)

input_list = []
head = create_linked_list_better(input_list)
test_function(input_list, head)
traverse_linkedlist(head)

# FINAL CORRECT BETTER LINKED LIST WITH  for loop
def create_better_linked_list(input_list):
	head = None
	tail = None
	for elem in input_list:
		if head is None:
			head = Node(elem)
			tail = head
		else:
			tail.next = Node(elem)
			tail = tail.next
	return head

print("better linked list for loop")
input_list = [1, 2, 3, 4, 5, 6]
head = create_better_linked_list(input_list)
test_function(input_list, head)
traverse_linkedlist(head)

input_list = [1]
head = create_better_linked_list(input_list)
test_function(input_list, head)
traverse_linkedlist(head)

input_list = []
head = create_better_linked_list(input_list)
test_function(input_list, head)
traverse_linkedlist(head)
