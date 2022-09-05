.. image:: https://raw.githubusercontent.com/rbturnbull/corgi/main/docs/images/corgi-banner.svg

.. start-badges

|testing badge| |coverage badge| |docs badge| |black badge| |git3moji badge| |torchapp badge|

.. |testing badge| image:: https://github.com/rbturnbull/corgi/actions/workflows/testing.yml/badge.svg
    :target: https://github.com/rbturnbull/corgi/actions

.. |docs badge| image:: https://github.com/rbturnbull/corgi/actions/workflows/docs.yml/badge.svg
    :target: https://rbturnbull.github.io/corgi
    
.. |black badge| image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    
.. |coverage badge| image:: https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/rbturnbull/ee1b52dd314d6441e0aabc0e1e50dc2c/raw/coverage-badge.json
    :target: https://rbturnbull.github.io/corgi/coverage/

.. |git3moji badge| image:: https://img.shields.io/badge/git3moji-%E2%9A%A1%EF%B8%8F%F0%9F%90%9B%F0%9F%93%BA%F0%9F%91%AE%F0%9F%94%A4-fffad8.svg
    :target: https://robinpokorny.github.io/git3moji/

.. |torchapp badge| image:: https://img.shields.io/badge/MLOps-torchapp-fuchsia.svg
    :target: https://github.com/rbturnbull/torchapp
        
.. end-badges

.. start-quickstart

Installation
============

The software can be installed using ``pip``

.. code-block:: bash

    pip install git+https://github.com/rbturnbull/corgi.git

.. note ::

    Soon corgi will be able to be installed using PyPI.


Usage
============

To make predictions, run the ``corgi`` command line tool:

.. code-block:: bash

    corgi infer --pretrained <model pickle file> --fasta <input fasta file> --output-csv <results>

.. note ::

    Soon models will be able to be automatically downloaded by corgi.


For more information about the other options, see the help with:

.. code-block:: bash

    corgi --help


.. end-quickstart


Credits
==================================

* Robert Turnbull <robert.turnbull@unimelb.edu.au>
* Created using torchapp (https://github.com/rbturnbull/torchapp)

