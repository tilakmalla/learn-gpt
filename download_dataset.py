"""
Alternative Datasets for GPT Training
=====================================
Choose a dataset you can actually understand and evaluate!

Usage:
    python download_dataset.py --list           # See all options
    python download_dataset.py --dataset python # Download Python code
    python download_dataset.py --dataset physics # Download physics text
"""

import os
import urllib.request
import argparse

DATASETS = {
    'shakespeare': {
        'url': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        'description': 'Shakespeare plays (~1MB) - Old English, dialogue format',
        'example_prompt': 'ROMEO:',
        'what_to_look_for': 'Character names, dialogue structure, iambic pentameter',
    },
    'python': {
        'url': 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',  # Placeholder - we'll generate
        'description': 'Python code (~2MB) - Functions, classes, comments',
        'example_prompt': 'def ',
        'what_to_look_for': 'Valid Python syntax, proper indentation, function names',
        'custom_generator': True,
    },
    'physics': {
        'url': None,
        'description': 'Physics concepts and explanations (~1MB)',
        'example_prompt': 'Newton',
        'what_to_look_for': 'Physics terms, equations described in words, concepts',
        'custom_generator': True,
    },
    'simple_english': {
        'url': None,
        'description': 'Simple English text - easy to read and evaluate',
        'example_prompt': 'The ',
        'what_to_look_for': 'Grammatically correct sentences, common words',
        'custom_generator': True,
    },
    'lyrics': {
        'url': None,
        'description': 'Song lyrics structure',
        'example_prompt': '[Verse 1]',
        'what_to_look_for': 'Rhyming patterns, verse/chorus structure',
        'custom_generator': True,
    },
}


def download_file(url: str, filepath: str):
    """Download a file from URL."""
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, filepath)
    size = os.path.getsize(filepath)
    print(f"Downloaded: {filepath} ({size:,} bytes)")


def generate_physics_dataset(filepath: str):
    """Generate a physics concepts dataset."""
    
    # Physics text covering various topics
    physics_text = '''
# Classical Mechanics

Newton's First Law states that an object at rest stays at rest, and an object in motion stays in motion with the same speed and direction, unless acted upon by an unbalanced force. This is also known as the law of inertia.

Newton's Second Law describes how force equals mass times acceleration, written as F = ma. When you push an object, it accelerates in the direction of the force. A heavier object requires more force to achieve the same acceleration.

Newton's Third Law states that for every action, there is an equal and opposite reaction. When you push against a wall, the wall pushes back against you with equal force.

Momentum is defined as mass times velocity, p = mv. In a closed system, total momentum is conserved. When two objects collide, their total momentum before equals their total momentum after.

Kinetic energy is the energy of motion, calculated as one-half times mass times velocity squared. A moving car has kinetic energy. A faster car has more kinetic energy. A heavier car also has more kinetic energy.

Potential energy is stored energy based on position. Gravitational potential energy equals mass times gravity times height. A ball held high has potential energy that converts to kinetic energy as it falls.

Work is done when a force moves an object through a distance. Work equals force times distance times the cosine of the angle between them. Lifting a box does positive work against gravity.

Power is the rate of doing work, measured in watts. One watt equals one joule per second. A 100-watt light bulb uses 100 joules of energy every second.

# Thermodynamics

Temperature measures the average kinetic energy of particles in a substance. Higher temperature means particles move faster. Absolute zero is the lowest possible temperature where all particle motion stops.

Heat is energy transferred due to temperature difference. Heat flows from hot objects to cold objects until thermal equilibrium is reached. This is described by the zeroth law of thermodynamics.

The first law of thermodynamics states that energy cannot be created or destroyed, only transformed. The total energy of an isolated system remains constant. This is conservation of energy.

The second law of thermodynamics states that entropy of an isolated system always increases. Heat flows spontaneously from hot to cold, never the reverse. Perpetual motion machines are impossible.

Entropy is a measure of disorder in a system. A shuffled deck has higher entropy than an ordered deck. The universe tends toward maximum entropy over time.

Specific heat capacity is the energy needed to raise one kilogram of a substance by one degree. Water has a high specific heat, which is why oceans moderate climate.

# Waves and Oscillations

A wave is a disturbance that transfers energy without transferring matter. Ocean waves transfer energy across the water, but the water molecules stay in roughly the same place.

Wavelength is the distance between consecutive wave peaks. Frequency is the number of waves passing a point per second. Wave speed equals wavelength times frequency.

Sound is a longitudinal wave where particles vibrate parallel to the direction of wave travel. Sound requires a medium like air or water to propagate. Sound cannot travel through vacuum.

Light is an electromagnetic wave that can travel through vacuum. The speed of light in vacuum is approximately 300 million meters per second. Nothing can travel faster than light.

Interference occurs when two waves meet. Constructive interference happens when peaks align, creating a larger wave. Destructive interference happens when a peak meets a trough, canceling out.

Resonance occurs when a system is driven at its natural frequency. A wine glass can shatter when a singer hits the right note. Bridges must be designed to avoid resonance with wind.

# Electricity and Magnetism

Electric charge comes in two types, positive and negative. Like charges repel, opposite charges attract. Electrons carry negative charge, protons carry positive charge.

Electric current is the flow of electric charge, measured in amperes. One ampere equals one coulomb of charge per second. Current flows from high potential to low potential.

Voltage is electric potential difference, measured in volts. A 9-volt battery has a potential difference of 9 volts between its terminals. Voltage drives current through a circuit.

Resistance opposes current flow, measured in ohms. Ohm's law states voltage equals current times resistance. A higher resistance means less current for the same voltage.

Power in a circuit equals voltage times current. A 100-watt device on 120 volts draws about 0.83 amperes. Power equals current squared times resistance.

Magnetic fields are produced by moving charges and magnets. A current-carrying wire creates a magnetic field around it. Earth's magnetic field protects us from solar radiation.

Electromagnetic induction occurs when a changing magnetic field creates an electric field. Generators use this to produce electricity. Transformers use induction to change voltage levels.

# Modern Physics

Einstein's special relativity shows that space and time are connected as spacetime. Time passes slower for fast-moving objects. Nothing can exceed the speed of light.

The famous equation E equals mc squared shows mass and energy are equivalent. A small amount of mass contains enormous energy. Nuclear reactions convert tiny mass differences into huge energy.

Quantum mechanics describes physics at atomic scales. Particles behave like waves and waves behave like particles. Electrons exist in probability clouds around nuclei.

The uncertainty principle states you cannot precisely know both position and momentum simultaneously. Measuring one disturbs the other. This is fundamental, not a measurement limitation.

Photons are particles of light with energy proportional to frequency. Higher frequency light has more energetic photons. Ultraviolet has more energy than infrared.

The photoelectric effect shows light behaves as particles. Light below a threshold frequency cannot eject electrons regardless of intensity. This proved light has particle nature.

Atoms consist of a nucleus containing protons and neutrons, surrounded by electrons. Protons have positive charge, electrons have negative charge, neutrons are neutral.

Nuclear fission splits heavy atoms into lighter ones, releasing energy. Uranium-235 can undergo fission when struck by a neutron. Nuclear power plants use controlled fission.

Nuclear fusion combines light atoms into heavier ones, releasing energy. The sun produces energy by fusing hydrogen into helium. Fusion requires extremely high temperatures.

Radioactive decay is the spontaneous transformation of unstable nuclei. Half-life is the time for half the atoms to decay. Carbon-14 dating uses radioactive decay to determine age.

# Gravity and Orbits

Gravity is an attractive force between all masses. The gravitational force is proportional to both masses and inversely proportional to distance squared.

Orbits result from the balance between gravity and inertia. A satellite in orbit is constantly falling toward Earth but moving forward fast enough to keep missing.

Escape velocity is the minimum speed needed to escape a gravitational field. Earth's escape velocity is about 11 kilometers per second. Rockets must reach this speed to leave Earth.

Gravitational potential energy between two masses is negative and increases toward zero as distance increases. It takes energy to separate gravitationally bound objects.

Tides are caused by the differential gravitational pull of the Moon and Sun. The side of Earth closer to the Moon experiences stronger gravity than the far side.

General relativity describes gravity as the curvature of spacetime caused by mass. Massive objects bend the fabric of spacetime. Light follows curved paths near massive objects.

Black holes form when massive stars collapse. Their gravity is so strong that nothing, not even light, can escape once past the event horizon.

Gravitational waves are ripples in spacetime caused by accelerating masses. They were first detected in 2015 from merging black holes. This confirmed Einstein's prediction.

# Fluid Mechanics

Pressure is force per unit area. Atmospheric pressure at sea level is about 101,000 pascals. Pressure increases with depth in a fluid.

Buoyancy is the upward force on an object in a fluid. An object floats if its density is less than the fluid's density. Ships float because their average density is less than water.

Bernoulli's principle states that faster-moving fluid has lower pressure. Airplane wings use this to generate lift. The curved top surface causes air to move faster, reducing pressure.

Viscosity is a fluid's resistance to flow. Honey has high viscosity, water has low viscosity. Viscous forces dissipate energy in flowing fluids.

Turbulence is chaotic fluid flow at high speeds. Reynolds number predicts when flow becomes turbulent. Smooth flow is called laminar flow.

# Optics

Reflection occurs when light bounces off a surface. The angle of incidence equals the angle of reflection. Mirrors use reflection to form images.

Refraction is the bending of light when it passes between media. Light slows down in denser media and bends toward the normal. This is why pools appear shallower than they are.

Lenses use refraction to focus or spread light. Convex lenses converge light to a focal point. Concave lenses diverge light outward.

Total internal reflection occurs when light cannot escape a denser medium. This is how fiber optic cables transmit light. Diamonds sparkle due to total internal reflection.

The electromagnetic spectrum includes radio waves, microwaves, infrared, visible light, ultraviolet, X-rays, and gamma rays. They differ only in frequency and wavelength.

Color perception depends on wavelength of visible light. Red has the longest wavelength, violet the shortest. White light contains all colors mixed together.

Polarization describes the orientation of light wave oscillation. Polarized sunglasses block certain orientations, reducing glare. LCD screens use polarized light.

# Atomic and Nuclear Physics

Electrons occupy discrete energy levels in atoms. They can jump between levels by absorbing or emitting photons. Each element has a unique emission spectrum.

Isotopes are atoms of the same element with different numbers of neutrons. Carbon-12 and carbon-14 are isotopes. Some isotopes are stable, others are radioactive.

The strong nuclear force holds protons and neutrons together in the nucleus. It is much stronger than electromagnetic repulsion but only acts at very short range.

The weak nuclear force is responsible for radioactive beta decay. A neutron can transform into a proton by emitting an electron and antineutrino.

Particle accelerators speed up particles to study fundamental physics. The Large Hadron Collider discovered the Higgs boson. Higher energies probe smaller distance scales.

Antimatter consists of particles with opposite properties to normal matter. Positrons are antielectrons. When matter meets antimatter, they annihilate into pure energy.

# Astrophysics

Stars form from collapsing clouds of gas and dust. Gravity pulls material together until fusion ignites in the core. Our sun has been fusing hydrogen for 4.6 billion years.

Main sequence stars fuse hydrogen into helium in their cores. The outward pressure from fusion balances inward gravitational pull. Stars spend most of their lives on the main sequence.

Red giants form when stars exhaust their core hydrogen. The core contracts and heats while outer layers expand and cool. Our sun will become a red giant in about 5 billion years.

Supernovae are explosive deaths of massive stars. The core collapses suddenly, and the rebounding material creates heavy elements. Many elements in your body came from supernovae.

Neutron stars are incredibly dense remnants of supernovae. A teaspoon of neutron star material would weigh billions of tons. Pulsars are rapidly rotating neutron stars.

White dwarfs are Earth-sized stellar remnants supported by electron degeneracy pressure. They slowly cool over billions of years. Our sun will end as a white dwarf.

The Hertzsprung-Russell diagram plots stars by temperature and luminosity. Main sequence stars form a diagonal band. Giants and white dwarfs occupy other regions.

Galaxies are vast collections of stars, gas, and dark matter. The Milky Way contains hundreds of billions of stars. Galaxies cluster together under mutual gravity.

Dark matter is invisible matter detected only through its gravitational effects. It makes up about 27 percent of the universe. Its nature remains unknown.

Dark energy is causing the expansion of the universe to accelerate. It makes up about 68 percent of the universe. Its nature is even more mysterious than dark matter.

The Big Bang theory describes the origin of the universe from an extremely hot, dense state about 13.8 billion years ago. The universe has been expanding ever since.

Cosmic microwave background radiation is the afterglow of the Big Bang. It fills all of space at a temperature of 2.7 kelvin. Its patterns reveal the early universe's structure.

'''
    
    # Repeat the content to make it larger (aim for ~1MB)
    full_text = physics_text * 8  # ~1MB
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"Generated physics dataset: {filepath}")
    print(f"Size: {len(full_text):,} characters")


def generate_python_dataset(filepath: str):
    """Generate a Python code dataset."""
    
    python_code = '''
# Basic Python Functions

def add_numbers(a, b):
    """Add two numbers together and return the result."""
    result = a + b
    return result

def multiply_numbers(x, y):
    """Multiply two numbers and return the product."""
    return x * y

def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    if len(numbers) == 0:
        return 0
    total = sum(numbers)
    average = total / len(numbers)
    return average

def find_maximum(numbers):
    """Find the maximum value in a list."""
    if not numbers:
        return None
    max_value = numbers[0]
    for num in numbers:
        if num > max_value:
            max_value = num
    return max_value

def find_minimum(numbers):
    """Find the minimum value in a list."""
    if not numbers:
        return None
    min_value = numbers[0]
    for num in numbers:
        if num < min_value:
            min_value = num
    return min_value

def count_occurrences(items, target):
    """Count how many times target appears in items."""
    count = 0
    for item in items:
        if item == target:
            count += 1
    return count

def reverse_string(text):
    """Reverse a string and return it."""
    reversed_text = ""
    for char in text:
        reversed_text = char + reversed_text
    return reversed_text

def is_palindrome(text):
    """Check if a string is a palindrome."""
    text = text.lower()
    reversed_text = reverse_string(text)
    return text == reversed_text

def factorial(n):
    """Calculate the factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    return fibonacci(n - 1) + fibonacci(n - 2)

def is_prime(n):
    """Check if a number is prime."""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

def get_primes(limit):
    """Return a list of prime numbers up to limit."""
    primes = []
    for num in range(2, limit + 1):
        if is_prime(num):
            primes.append(num)
    return primes

def bubble_sort(numbers):
    """Sort a list using bubble sort algorithm."""
    nums = numbers.copy()
    n = len(nums)
    for i in range(n):
        for j in range(0, n - i - 1):
            if nums[j] > nums[j + 1]:
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
    return nums

def binary_search(sorted_list, target):
    """Search for target in a sorted list using binary search."""
    left = 0
    right = len(sorted_list) - 1
    while left <= right:
        mid = (left + right) // 2
        if sorted_list[mid] == target:
            return mid
        elif sorted_list[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1


# Classes and Object-Oriented Programming

class Rectangle:
    """A class representing a rectangle."""
    
    def __init__(self, width, height):
        """Initialize the rectangle with width and height."""
        self.width = width
        self.height = height
    
    def area(self):
        """Calculate the area of the rectangle."""
        return self.width * self.height
    
    def perimeter(self):
        """Calculate the perimeter of the rectangle."""
        return 2 * (self.width + self.height)
    
    def is_square(self):
        """Check if the rectangle is a square."""
        return self.width == self.height


class Circle:
    """A class representing a circle."""
    
    def __init__(self, radius):
        """Initialize the circle with a radius."""
        self.radius = radius
        self.pi = 3.14159
    
    def area(self):
        """Calculate the area of the circle."""
        return self.pi * self.radius ** 2
    
    def circumference(self):
        """Calculate the circumference of the circle."""
        return 2 * self.pi * self.radius
    
    def diameter(self):
        """Return the diameter of the circle."""
        return 2 * self.radius


class BankAccount:
    """A class representing a bank account."""
    
    def __init__(self, owner, balance=0):
        """Initialize account with owner and optional balance."""
        self.owner = owner
        self.balance = balance
    
    def deposit(self, amount):
        """Deposit money into the account."""
        if amount > 0:
            self.balance += amount
            return True
        return False
    
    def withdraw(self, amount):
        """Withdraw money from the account."""
        if amount > 0 and amount <= self.balance:
            self.balance -= amount
            return True
        return False
    
    def get_balance(self):
        """Return the current balance."""
        return self.balance


class Stack:
    """A class implementing a stack data structure."""
    
    def __init__(self):
        """Initialize an empty stack."""
        self.items = []
    
    def push(self, item):
        """Add an item to the top of the stack."""
        self.items.append(item)
    
    def pop(self):
        """Remove and return the top item."""
        if not self.is_empty():
            return self.items.pop()
        return None
    
    def peek(self):
        """Return the top item without removing it."""
        if not self.is_empty():
            return self.items[-1]
        return None
    
    def is_empty(self):
        """Check if the stack is empty."""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in the stack."""
        return len(self.items)


class Queue:
    """A class implementing a queue data structure."""
    
    def __init__(self):
        """Initialize an empty queue."""
        self.items = []
    
    def enqueue(self, item):
        """Add an item to the back of the queue."""
        self.items.append(item)
    
    def dequeue(self):
        """Remove and return the front item."""
        if not self.is_empty():
            return self.items.pop(0)
        return None
    
    def front(self):
        """Return the front item without removing it."""
        if not self.is_empty():
            return self.items[0]
        return None
    
    def is_empty(self):
        """Check if the queue is empty."""
        return len(self.items) == 0
    
    def size(self):
        """Return the number of items in the queue."""
        return len(self.items)


class LinkedListNode:
    """A node in a linked list."""
    
    def __init__(self, data):
        """Initialize node with data."""
        self.data = data
        self.next = None


class LinkedList:
    """A singly linked list implementation."""
    
    def __init__(self):
        """Initialize an empty linked list."""
        self.head = None
    
    def append(self, data):
        """Add a node to the end of the list."""
        new_node = LinkedListNode(data)
        if self.head is None:
            self.head = new_node
            return
        current = self.head
        while current.next:
            current = current.next
        current.next = new_node
    
    def prepend(self, data):
        """Add a node to the beginning of the list."""
        new_node = LinkedListNode(data)
        new_node.next = self.head
        self.head = new_node
    
    def delete(self, data):
        """Delete the first node with the given data."""
        if self.head is None:
            return
        if self.head.data == data:
            self.head = self.head.next
            return
        current = self.head
        while current.next:
            if current.next.data == data:
                current.next = current.next.next
                return
            current = current.next
    
    def find(self, data):
        """Find a node with the given data."""
        current = self.head
        while current:
            if current.data == data:
                return current
            current = current.next
        return None
    
    def to_list(self):
        """Convert linked list to Python list."""
        result = []
        current = self.head
        while current:
            result.append(current.data)
            current = current.next
        return result


# File Operations

def read_file(filename):
    """Read and return contents of a file."""
    with open(filename, 'r') as file:
        contents = file.read()
    return contents

def write_file(filename, content):
    """Write content to a file."""
    with open(filename, 'w') as file:
        file.write(content)

def append_to_file(filename, content):
    """Append content to a file."""
    with open(filename, 'a') as file:
        file.write(content)

def count_lines(filename):
    """Count the number of lines in a file."""
    with open(filename, 'r') as file:
        lines = file.readlines()
    return len(lines)

def count_words(filename):
    """Count the number of words in a file."""
    with open(filename, 'r') as file:
        contents = file.read()
    words = contents.split()
    return len(words)


# List Operations

def remove_duplicates(items):
    """Remove duplicate items from a list."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def flatten_list(nested_list):
    """Flatten a nested list into a single list."""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result

def chunk_list(items, chunk_size):
    """Split a list into chunks of specified size."""
    chunks = []
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

def zip_lists(list1, list2):
    """Combine two lists into list of tuples."""
    result = []
    min_length = min(len(list1), len(list2))
    for i in range(min_length):
        result.append((list1[i], list2[i]))
    return result


# String Operations

def count_vowels(text):
    """Count the number of vowels in a string."""
    vowels = "aeiouAEIOU"
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count

def count_consonants(text):
    """Count the number of consonants in a string."""
    consonants = "bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ"
    count = 0
    for char in text:
        if char in consonants:
            count += 1
    return count

def remove_whitespace(text):
    """Remove all whitespace from a string."""
    return "".join(text.split())

def capitalize_words(text):
    """Capitalize the first letter of each word."""
    words = text.split()
    capitalized = [word.capitalize() for word in words]
    return " ".join(capitalized)

def is_anagram(word1, word2):
    """Check if two words are anagrams."""
    return sorted(word1.lower()) == sorted(word2.lower())


# Math Operations

def gcd(a, b):
    """Calculate the greatest common divisor."""
    while b:
        a, b = b, a % b
    return a

def lcm(a, b):
    """Calculate the least common multiple."""
    return abs(a * b) // gcd(a, b)

def power(base, exponent):
    """Calculate base raised to exponent."""
    if exponent == 0:
        return 1
    if exponent < 0:
        return 1 / power(base, -exponent)
    result = 1
    for _ in range(exponent):
        result *= base
    return result

def absolute_value(n):
    """Return the absolute value of a number."""
    if n < 0:
        return -n
    return n

def clamp(value, minimum, maximum):
    """Clamp a value between minimum and maximum."""
    if value < minimum:
        return minimum
    if value > maximum:
        return maximum
    return value


# Dictionary Operations

def merge_dicts(dict1, dict2):
    """Merge two dictionaries."""
    result = dict1.copy()
    result.update(dict2)
    return result

def invert_dict(d):
    """Swap keys and values in a dictionary."""
    return {v: k for k, v in d.items()}

def filter_dict(d, predicate):
    """Filter dictionary by predicate function on values."""
    return {k: v for k, v in d.items() if predicate(v)}

def get_keys_by_value(d, value):
    """Get all keys that have a specific value."""
    return [k for k, v in d.items() if v == value]


# Error Handling

def safe_divide(a, b):
    """Safely divide two numbers."""
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        return None

def safe_int_convert(text):
    """Safely convert string to integer."""
    try:
        return int(text)
    except ValueError:
        return None

def safe_list_access(items, index):
    """Safely access list element by index."""
    try:
        return items[index]
    except IndexError:
        return None


# Recursion Examples

def sum_digits(n):
    """Sum all digits in a number recursively."""
    n = abs(n)
    if n < 10:
        return n
    return n % 10 + sum_digits(n // 10)

def count_digits(n):
    """Count digits in a number recursively."""
    n = abs(n)
    if n < 10:
        return 1
    return 1 + count_digits(n // 10)

def reverse_list(items):
    """Reverse a list recursively."""
    if len(items) <= 1:
        return items
    return [items[-1]] + reverse_list(items[:-1])

def binary_to_decimal(binary_str):
    """Convert binary string to decimal recursively."""
    if len(binary_str) == 0:
        return 0
    if binary_str[-1] == '1':
        return 1 + 2 * binary_to_decimal(binary_str[:-1])
    return 2 * binary_to_decimal(binary_str[:-1])


# Main execution examples

if __name__ == "__main__":
    # Test basic functions
    print("Testing add_numbers:", add_numbers(5, 3))
    print("Testing factorial:", factorial(5))
    print("Testing is_prime:", is_prime(17))
    
    # Test classes
    rect = Rectangle(4, 5)
    print("Rectangle area:", rect.area())
    
    circle = Circle(3)
    print("Circle area:", circle.area())
    
    # Test data structures
    stack = Stack()
    stack.push(1)
    stack.push(2)
    stack.push(3)
    print("Stack pop:", stack.pop())
    
    # Test algorithms
    numbers = [64, 34, 25, 12, 22, 11, 90]
    sorted_numbers = bubble_sort(numbers)
    print("Sorted:", sorted_numbers)

'''
    
    # Repeat to make larger
    full_code = python_code * 5  # ~1MB
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_code)
    
    print(f"Generated Python dataset: {filepath}")
    print(f"Size: {len(full_code):,} characters")


def generate_simple_english(filepath: str):
    """Generate simple English text dataset."""
    
    simple_text = '''
The sun rises in the east every morning. It brings light and warmth to the world. People wake up and start their day. Birds sing in the trees. The sky turns from dark to bright blue.

A dog is a loyal friend. Dogs love to play and run. They wag their tails when they are happy. Dogs can learn many tricks. They protect their families and homes.

Water is essential for life. We drink water every day. Plants need water to grow. Rivers and lakes hold fresh water. The ocean is filled with salt water.

Trees are important for our planet. They give us oxygen to breathe. Trees provide shade on hot days. Many animals live in trees. Wood from trees builds houses.

The moon shines at night. It reflects light from the sun. The moon changes shape each month. We call these phases of the moon. Some nights the moon is full and bright.

Books contain knowledge and stories. Reading helps us learn new things. Libraries have many books to borrow. Some books are fiction, some are facts. Reading improves our vocabulary.

Music makes people happy. There are many types of music. Some music is fast, some is slow. People sing and play instruments. Music is found in every culture.

Food gives us energy. We eat breakfast, lunch, and dinner. Fruits and vegetables are healthy. Cooking is a useful skill. Sharing food brings people together.

Exercise keeps us healthy. Walking is simple exercise. Running makes the heart strong. Swimming uses many muscles. Playing sports is fun exercise.

Sleep is important for health. Most people need eight hours of sleep. Dreams happen during sleep. A good bed helps us rest. Sleeping well improves memory.

Computers are powerful tools. They help us work and learn. The internet connects computers worldwide. Email lets us send messages instantly. Many jobs require computer skills.

Cars help us travel far. They have four wheels and an engine. Some cars use gasoline, others use electricity. Traffic lights control car movement. Seat belts keep us safe.

Weather changes from day to day. Rain brings water to the land. Snow falls when it is very cold. Wind moves clouds across the sky. Sunny days are warm and bright.

Mountains are very tall landforms. They are covered with rocks and snow. Climbing mountains is challenging. Many rivers start in mountains. Views from mountains are beautiful.

Friends make life better. Good friends listen to each other. Friends help during difficult times. Spending time with friends is enjoyable. Trust is important in friendship.

Schools teach many subjects. Students learn math and science. Reading and writing are fundamental. History tells us about the past. Art and music develop creativity.

Doctors help sick people. Nurses care for patients. Hospitals have medical equipment. Medicine treats many illnesses. Staying healthy prevents disease.

Cities have many buildings. People live and work in cities. Streets connect different areas. Public transportation moves people around. Parks provide green space in cities.

Farms grow our food. Farmers plant seeds and harvest crops. Animals like cows and chickens live on farms. Tractors help with farm work. Fresh food comes from local farms.

The ocean is vast and deep. Fish and whales live in the ocean. Waves crash on the shore. Ships sail across the ocean. The beach is where land meets sea.

Stars twinkle in the night sky. They are very far away. Our sun is the closest star. Constellations are patterns of stars. Scientists study stars with telescopes.

Gardens grow flowers and vegetables. People plant seeds in spring. Watering helps plants grow. Weeds must be removed regularly. Harvesting happens in summer and fall.

Money is used to buy things. People earn money by working. Banks keep money safe. Saving money is a good habit. Spending wisely is important.

Phones let us talk to anyone. Cell phones are portable. We can send text messages. Apps provide many functions. Phones have cameras for photos.

Holidays bring families together. People celebrate with food and gifts. Different cultures have different holidays. Traditions are passed down through generations. Memories are made during holidays.

Sports are games people play. Teams compete against each other. Winning requires skill and practice. Fans cheer for their favorite teams. Exercise from sports keeps us fit.

Animals live everywhere on Earth. Some animals are wild, some are pets. Animals need food and water. Different animals eat different things. Protecting animals preserves nature.

Fire provides heat and light. Humans discovered fire long ago. Cooking food requires fire. Fires must be controlled carefully. Firefighters put out dangerous fires.

Ice is frozen water. Ice is cold and slippery. Ice cubes cool our drinks. Glaciers are huge sheets of ice. Ice melts when it gets warm.

Wind is moving air. Strong wind can push things around. Wind turbines make electricity. Kites fly in the wind. Trees bend when wind blows hard.

Colors make the world beautiful. Red, blue, and yellow are primary colors. Mixing colors creates new colors. Rainbows show many colors together. Artists use colors to create art.

Time moves forward constantly. We measure time with clocks. Seconds, minutes, and hours pass. Days become weeks and months. Years mark our age and history.

Maps show us locations. They help us find our way. Countries and cities are on maps. GPS uses maps for navigation. Old maps were drawn by hand.

Languages let us communicate. Many languages exist worldwide. Learning languages opens doors. Writing records language forever. Speaking connects people instantly.

Electricity powers our modern world. Light bulbs use electricity. Appliances run on electricity. Power plants generate electricity. Saving electricity helps the environment.

Bridges cross over obstacles. They connect separated areas. Some bridges are very long. Different designs serve different needs. Engineers design safe bridges.

Airplanes fly through the sky. They transport people and goods. Pilots fly the airplanes. Airports are where planes land. Flying is the fastest way to travel far.

Seasons change throughout the year. Spring brings new growth. Summer is warm and sunny. Autumn leaves fall from trees. Winter brings cold and snow.

Rivers flow from mountains to oceans. They provide fresh water. Fish swim in rivers. Boats travel on rivers. Rivers shape the land over time.

Clouds float high in the sky. They are made of water droplets. Dark clouds bring rain. White fluffy clouds are common. Cloud shapes change constantly.

Rocks come in many forms. Some rocks are very old. Fossils are found in rocks. Mountains are made of rock. Rocks can be smooth or rough.

Insects are small creatures. Bees make honey. Butterflies have colorful wings. Ants work together in colonies. Some insects can fly.

Gravity pulls things down. Everything falls toward Earth. Gravity keeps us on the ground. The moon's gravity causes tides. Astronauts float without strong gravity.

Numbers help us count and measure. Math uses numbers for calculations. Big numbers count large amounts. Negative numbers are below zero. Numbers are found everywhere.

Scientists discover new knowledge. They ask questions and test ideas. Experiments prove or disprove theories. Science explains how things work. New discoveries change our understanding.

'''
    
    # Repeat to get ~1MB
    full_text = simple_text * 15
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(full_text)
    
    print(f"Generated simple English dataset: {filepath}")
    print(f"Size: {len(full_text):,} characters")


def list_datasets():
    """List all available datasets."""
    print("\n" + "="*70)
    print(" Available Datasets")
    print("="*70)
    
    for name, info in DATASETS.items():
        print(f"\n{name.upper()}")
        print(f"  Description: {info['description']}")
        print(f"  Example prompt: '{info['example_prompt']}'")
        print(f"  What to look for: {info['what_to_look_for']}")
    
    print("\n" + "-"*70)
    print("To download, run:")
    print("  python download_dataset.py --dataset <name>")
    print("\nRecommended for physics background: python, physics, simple_english")


def download_dataset(name: str):
    """Download or generate a dataset."""
    os.makedirs('data', exist_ok=True)
    filepath = f'data/input.txt'
    
    if name not in DATASETS:
        print(f"Unknown dataset: {name}")
        print("Available:", list(DATASETS.keys()))
        return
    
    info = DATASETS[name]
    
    print(f"\nSetting up dataset: {name}")
    print(f"Description: {info['description']}")
    
    if name == 'physics':
        generate_physics_dataset(filepath)
    elif name == 'python':
        generate_python_dataset(filepath)
    elif name == 'simple_english':
        generate_simple_english(filepath)
    elif info.get('url'):
        download_file(info['url'], filepath)
    else:
        print(f"Dataset {name} is not yet available.")
        return
    
    print(f"\n✅ Dataset ready at: {filepath}")
    print(f"\nExample prompts to try after training:")
    print(f"  python generate.py -p '{info['example_prompt']}' -n 200")
    print(f"\nWhat to look for in output:")
    print(f"  {info['what_to_look_for']}")


def main():
    parser = argparse.ArgumentParser(description='Download training datasets')
    parser.add_argument('--list', action='store_true', help='List available datasets')
    parser.add_argument('--dataset', '-d', type=str, help='Dataset to download')
    
    args = parser.parse_args()
    
    if args.list or (not args.dataset):
        list_datasets()
    elif args.dataset:
        download_dataset(args.dataset)


if __name__ == "__main__":
    main()
