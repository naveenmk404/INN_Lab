{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyN8fyGKhNgWvngBbHog1w3r",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/naveenmk404/INN_Lab/blob/main/Program_1.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import numpy as np\n",
        "\n",
        "class Neuron:\n",
        "    def __init__(self,input_size,activation_function=\"sigmoid\"):\n",
        "        self.weights = np.random.rand(input_size)\n",
        "        self.bias = np.random.rand()\n",
        "\n",
        "        if(activation_function == \"sigmoid\"):\n",
        "            self.activation_function = self.sigmoid\n",
        "            self.activation_derivative = self.sigmoid_derivative\n",
        "\n",
        "        elif(activation_function == \"step\"):\n",
        "            self.activation_function = self.step\n",
        "            self.activation_derivative = self.step_derivative\n",
        "\n",
        "        else :\n",
        "            raise ValueError(\"Invalid activation function \")\n",
        "\n",
        "    def sigmoid(self,x):\n",
        "        return 1/(1+np.exp(-x))\n",
        "\n",
        "    def sigmoid_derivative(self,x):\n",
        "        return x*(1-x)\n",
        "\n",
        "    def step(self,x):\n",
        "        return 1 if x>0 else 0\n",
        "\n",
        "    def step_derivative(self,x):\n",
        "        return 0\n",
        "\n",
        "    def forward(self,inputs):\n",
        "        weight_sum = np.dot(self.weights, inputs)+self.bias\n",
        "        output = self.activation_function(weight_sum)\n",
        "        return output\n",
        "\n",
        "input_size = 3\n",
        "inputs = np.random.rand(input_size)\n",
        "\n",
        "neuron = Neuron(input_size,activation_function=\"step\")\n",
        "\n",
        "output = neuron.forward(inputs)\n",
        "\n",
        "print(f\"inputs : {inputs}\")\n",
        "print(f\"Weights : {neuron.weights}\")\n",
        "print(f\"bias : {neuron.bias}\")\n",
        "\n",
        "print(f\"output : {output}\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1UDWpk93BRDB",
        "outputId": "9c28b1ca-9e48-455f-d597-fff534c75ed9"
      },
      "execution_count": 6,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "inputs : [0.06661563 0.35046464 0.47978481]\n",
            "Weights : [0.48257821 0.92404801 0.52479486]\n",
            "bias : 0.7119316301728422\n",
            "output : 1\n"
          ]
        }
      ]
    }
  ]
}