# Draft for our new, object-oriented API


class Registration:
    def __init__(
        self,
        *,
        loss,
        source_module,
        target_module,
        max_iter=100,
        tol=0.0,
    ):
        self.loss = loss
        self.source_module = source_module
        self.target_module = target_module
        self.max_iter = max_iter
        self.tol = tol

    def preprocess(self):
        pass

    def postprocess(self):
        pass

    def training_step(self):
        # "Shooting" step: apply the parametrics transformations to the source and target
        source_registered = self.source_module(self.source_)
        target_registered = self.target_module(self.target_)

        # "Matching" step: compute the coupling between the registered source and target
        # This usually corresponds to a closest point matching,
        # or to an optimal transport solve.
        self.loss.fit(source_registered, target_registered)

        # Turn the coupling into correspondences.
        # source_correspondences has the same shape and topology as self.source_,
        # with additional "precision" attributes that specify a point-wise loss function
        # and can be used to define a point-to-plane loss, for instance.
        source_correspondences = self.loss.source_correspondences_
        target_correspondences = self.loss.target_correspondences_

        # Use the correspondences to update the parametric transformations.
        # This is usually the compute-intensive part of the algorithm,
        # involving a linear solve, an SVD decomposition, etc.
        # step_size should be used to control the ratio between the source
        # and target updates, without overshooting.
        self.source_module.fit(source_correspondences, step_size=0.5)
        self.target_module.fit(target_correspondences, step_size=0.5)

    @property
    def has_converged(self):
        return False

    @property
    def training_schedule(self):
        for _it in range(self.max_iter):
            if self.has_converged:
                return

            # By default, we just yield self without any modification.
            # More complex training schedules may update the loss, the source and target
            # modules, etc. at each iteration.
            yield self

    def fit(self, source, target):

        self.source_ = source
        self.target_ = target

        self.preprocess()

        for current_problem in self.training_schedule:
            current_problem.training_step()

        self.postprocess()

        return self
