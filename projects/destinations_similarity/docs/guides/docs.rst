.. _docs:

Building Sphinx docs
---------------------------

.. warning::
    This documentation was built thinking about linux/mac operating systems or executions using WSL.


In addition to the local README, we decided to detail a few more things about the project and maybe in a more playful way to facilitate understanding.

There's not much mystery about building the documentation in HTML. We've already automated some things to make it easier. Generally speaking, Sphinx is responsible for creating a static page of HTML documentation using manually typed information or other information inserted into the developed code. These generated static pages are moved into a folder in a container running an NGINX image which hosts the documentation page.


To build the Docker image responsible for the documentation and start hosting the server, just run the command

.. warning::
    For this command to work it is necessary that in your location it is possible to run Makefile files and ensure that the working directory is inside projects/destinations_similarity

.. code-block::

    make open-docs

Once the command has been successfully executed, you can check with the command below if the container is running normally on your machine.

.. code-block::

    docker ps

the result should be

.. code-block::

    $ docker ps
    CONTAINER ID   IMAGE     COMMAND                  CREATED          STATUS          PORTS                                   NAMES
    84cb390d977f   nginx     "/docker-entrypoint.…"   36 seconds ago   Up 35 seconds   0.0.0.0:8080->80/tcp, :::8080->80/tcp   sphinx-nginx
