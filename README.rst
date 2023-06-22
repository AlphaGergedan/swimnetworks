==============================
SWIM: sample where it matters!
==============================

``swimnetworks`` implements the algorithm SWIM for sampling weights of neural networks.
The algorithm provides a way to quickly train neural networks on a CPU.
For more details on the theoretical background of the method, refer to our paper [1]_.

Installation
------------

To install the main package with the requirements, one needs to clone the repository and execute the follwoing command from the root folder:

.. code-block:: bash

    pip install .

Example
-------

Here is a small example of defining a sampled network:

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from swimnetworks import Dense, Linear

    steps = [
        (f"dense", Dense(layer_width=512, activation="tanh",
                         parameter_sampler="tanh",
                         random_seed=42)),
        ("linear", Linear(regularization_scale=1e-10))
    ]
    model = Pipeline(steps)

Then, one can use :code:`model.fit(X_train, y_train)` and :code:`model.transform(X_test)` to train and evaluate the model.
The numerical experiments from [1]_ can be found in a separate `repository`_.

Citation
--------

If you use the SWIM package in your research, please cite the following `paper`_:

.. [1] E\. Bolager, I. Burak, C. Datar, Q. Sun, F. Dietrich. Sampling weights of deep neural networks. arXiV, 2023.

.. _paper: https://arxiv.org/abs/XXX.XXXX

.. _repository: https://gitlab.com/felix.dietrich/swimnetworks-experiments
