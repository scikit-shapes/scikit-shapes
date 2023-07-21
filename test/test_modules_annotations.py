import inspect
from typing import Union, get_origin, get_args
import skshapes as sks


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
            if get_origin(reference_annotations[key]) is Union:
                # If the reference annotation is a Union, check that the annotation is the Union itself or one of its arguments
                # useful for example as Shape is defined as Union of specific shapes structures (PolyData, Image, etc.) and
                # some losses/morphings are limited to specific shapes
                # see : https://stackoverflow.com/questions/45957615/check-a-variable-against-union-type-at-runtime-in-python-3-6
                assert annotations[key] == reference_annotations[key] or annotations[
                    key
                ] in get_args(reference_annotations[key])
            else:
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


# Define the templates for the annotations of Loss and Morphing
loss_template = {
    "__call__": {
        "source": sks.Shape,
        "target": sks.Shape,
        "return": sks.FloatScalar,
    }
}

morphing_template = {
    "morph": {
        "shape": sks.Shape,
        "return_path": bool,
        "return_regularization": bool,
        "return": sks.morphing.utils.MorphingOutput,
    }
}


# Define the tests
def test_losses():
    check_module_annotations(
        module=sks.loss, template=loss_template, type=sks.loss.baseloss.BaseLoss
    )


def test_morphing():
    check_module_annotations(
        module=sks.morphing,
        template=morphing_template,
        type=sks.morphing.basemodel.BaseModel,
    )
