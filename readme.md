Neural Network Sorting Numbers
=========

This program trains a neural network to sort a set of 3-digit numbers. Did you know a neural network can sort? Neither did I, until I wrote this program!

Neural Network Setup
---

The 3-digit numbers are fed into the neural network by separating each digit as an input.
Digits are normalized to fall between 0-1 by dividing each digit by 9.
The number of inputs are equal to the number of digits in the numbers being sorted. For two 3-digit numbers, there would be 6 inputs.
The number of outputs are equal to the number of inputs. The output cooresponds to the actual digits, in sorted form.
Numbers in. Sorted numbers out.

Results
---

A neural network with two hidden layers of 25 neurons each (not including the output layer), trained for 6,000 iterations at a learning rate of 0.3, produces a training/test accuracy of 78%/74%.

Running It
---

```
node app
```

Generating Training/Test Data
---

Data can be generated by calling the generate() method. The first parameter is the number of rows to generate in the set. The second parameter is the number of numbers to sort. The third parameter is the number of digits per number.

```
console.log(NeuralNetworkManager.generate(5, 2, 3)); // Generate five examples of two 3-digit numbers. 

// Example generated data.
[ { input: [ 145, 657 ], output: [ 145, 657 ] },
  { input: [ 853, 719 ], output: [ 719, 853 ] },
  { input: [ 895, 278 ], output: [ 278, 895 ] },
  { input: [ 724, 910 ], output: [ 724, 910 ] },
  { input: [ 883, 970 ], output: [ 883, 970 ] } ]
```

After generating, the data will need to be formatted for input into the neural network. You can call the format() method to do this. The numbers will be separated by digit, so that each digit may be fed as an input into the neural network.

You can combine the two calls to generate and format by simply calling:

```
console.log(NeuralNetworkManager.generateFormatted(5, 2, 3));

// Example generated data.
[ { input: [ 1, 4, 5, 6, 5, 7 ], output: [ 1, 4, 5, 6, 5, 7 ] },
  { input: [ 8, 5, 3, 7, 1, 9 ], output: [ 7, 1, 9, 8, 5, 3 ] },
  { input: [ 8, 9, 5, 2, 7, 8 ], output: [ 2, 7, 8, 8, 9, 5 ] },
  { input: [ 7, 2, 4, 9, 1, 0 ], output: [ 7, 2, 4, 9, 1, 0 ] },
  { input: [ 8, 8, 3, 9, 7, 0 ], output: [ 8, 8, 3, 9, 7, 0 ] } ]
```

License
----

MIT

Author
----
Kory Becker
http://www.primaryobjects.com/kory-becker
