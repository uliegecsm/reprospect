Resource acquisition for profiling
==================================

`GitHub` self-hosted runners
----------------------------

Our use case is the following.
We have a single machine with a GPU that has 2 `GitHub` self-hosted runner services running.
In the general case (CPU and GPU unit testing), it is fine that both services run concurrently,
as performance does not matter.

However, some jobs --- such as GPU profiling --- *may* required
that they are owning the precious resource and no one else can use it.
For instance, `ncu` uses hardware counters that cannot be "shared".

However, we don't want to come back to a single service setup.
Moreover, even with a single service, proper resource acquisition might
still be needed (*e.g.* tests run in parallel).

We need a serialization mechanism.

`GitHub` solutions
~~~~~~~~~~~~~~~~~~

They aren't very suitable because it's always at the job level (either
runner labels or conurrency groups).

File-based locking
~~~~~~~~~~~~~~~~~~

The easiest implementation is to go for a file-based locking.
Each process would be guaranteed to "acquire the lock".

One problem tht quickly arised is that our jobs run in `Docker`
containers, so we'll need to mount a common volume to all jobs that *may*
need resource acquisition.

`SQL` database
~~~~~~~~~~~~~~

Since we are synchronizing on the same machine, we don't want to setup a service
like `Redis`.

The solution is based on transactions in a `SQL` database.

Compared to the file-based approach, the `SQL` approach eases the implementation
of a `status` method (log all locks) and `cleanup` (remove locks that expired).
