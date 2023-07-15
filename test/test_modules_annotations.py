import inspect
import sys

sys.path.append(sys.path[0][:-4])

print(sys.path)


def check_module_annotations(module, template, type):
    """Check that the type and the classes' annotations of the methods of a module are correct.
    More precisely, we check that all the classes in the module are subclasses of the given type,
    then we check that the methods of the template are in the class, and finally we check that the
    annotations of the methods are correct.

    Args:
        module : the module to check
        template (dict): a dictionary of the form {method_name: {annotation_name: annotation_type}}
        type : the desired type of the classes in the module
    """

    def check_annotations(annotations, reference_annotations):
        # Loop over the annotations
        for key in reference_annotations.keys():
            assert key in annotations.keys()
            assert issubclass(annotations[key], reference_annotations[key])

    # Find the classes in the module
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            # Check that the class is a subclass of the given type
            assert issubclass(obj, type)

            # Check that the methods of the template are in the class
            for method_name in template.keys():
                assert method_name in obj.__dict__.keys()

            # Loop over the methods of the class
            print("Inspecting annotations of class {}:".format(name))
            for method_name, method in inspect.getmembers(obj):
                if method_name in template.keys():
                    print("Method name: {}".format(method_name))
                    # get the annotations of the method
                    annotations = method.__annotations__
                    print("Annotations: {}".format(annotations))
                    print("Reference annotations: {}".format(template[method_name]))
                    # Check that the annotations are correct
                    check_annotations(annotations, template[method_name])


import skshapes.loss
import skshapes.morphing
import skshapes

# Define the templates for the annotations of Loss and Morphing
loss_template = {
    "__call__": {
        "source": skshapes.Shape,
        "target": skshapes.Shape,
        "return": skshapes.FloatScalar,
    }
}

morphing_template = {
    "morph": {
        "shape": skshapes.Shape,
        "return_path": bool,
        "return_regularization": bool,
        "return": skshapes.MorphingOutput,
    }
}


# Define the tests
def test_losses():
    check_module_annotations(
        module=skshapes.loss, template=loss_template, type=skshapes.Loss
    )


def test_morphing():
    check_module_annotations(
        module=skshapes.morphing, template=morphing_template, type=skshapes.Morphing
    )
