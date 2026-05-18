# Draft for our new, object-oriented API
from ..input_validation import one_and_only_one


class Registration:
    def __init__(
        self,
        *,
        coupling,
        source_module,
        target_module,
        optimizer="alternating",
        max_iter=100,
        tol=0.0,
    ):
        self.coupling = coupling
        self.source_module = source_module
        self.target_module = target_module
        self.optimizer = optimizer
        self.max_iter = max_iter
        self.tol = tol

    def preprocess(self):
        # N.B.: Under the hood, we may call custom setters
        self.source_module.base_shape_ = self.source_
        self.target_module.base_shape_ = self.target_

    def postprocess(self):
        pass

    def training_step(self):
        if self.optimizer == "alternating":
            # "Shooting" step: apply the parametrics transformations to the source and target
            source_registered = self.source_module.morphed_shape_
            target_registered = self.target_module.morphed_shape_

            # "Matching" step: compute the coupling between the registered source and target
            # This usually corresponds to a closest point matching,
            # or to an optimal transport solve.
            self.coupling.fit(source_registered, target_registered)

            # Turn the coupling into correspondences.
            # source_correspondences has the same shape and topology as self.source_,
            # with additional "precision" attributes that specify a point-wise loss function
            # and can be used to define a point-to-plane loss, for instance.
            source_correspondences = self.coupling.source_correspondences_
            target_correspondences = self.coupling.target_correspondences_

            # Use the correspondences to update the parametric transformations.
            # This is usually the compute-intensive part of the algorithm,
            # involving a linear solve, an SVD decomposition, etc.
            # step_size should be used to control the ratio between the source
            # and target updates, without overshooting.
            self.source_module.fit(source_correspondences, step_size=0.5)
            self.target_module.fit(target_correspondences, step_size=0.5)

        else:

            def closure():
                self.optimizer.zero_grad()
                source_registered = self.source_module.morphed_shape_
                target_registered = self.target_module.morphed_shape_
                self.coupling.fit(source_registered, target_registered)
                loss = self.penalty_
                loss.backward()
                return loss

            self.optimizer.step(closure)

    @property
    def has_converged_(self):
        return False

    @property
    def penalty_(self):
        return (
            self.coupling.penalty_
            + self.source_module.penalty_
            + self.target_module.penalty_
        )

    @property
    def training_schedule(self):
        for _it in range(self.max_iter):
            if self.has_converged_:
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

    @one_and_only_one(["source_signal", "target_signal"])
    def transfer(
        self,
        *,
        source_signal=None,
        target_signal=None,
        reg=None,
    ):
        if source_signal is not None:
            # Transfer from the source to the target
            return self.coupling.transfer(source_signal=source_signal, reg=reg)
        else:
            # Transfer from the target to the source
            return self.coupling.transfer(target_signal=target_signal, reg=reg)
