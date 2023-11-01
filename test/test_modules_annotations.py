import inspect
from typing import Union, get_origin, get_args
import skshapes as sks


def check_annotations(annotations, reference_annotations):
    # Loop over the annotations
    for key in reference_annotations.keys():
        assert key in annotations.keys()
        if get_origin(reference_annotations[key]) is Union:
            # If the reference annotation is a Union, check that the
            # annotation is the Union itself or one of its arguments
            # useful for example as Shape is defined as Union of specific
            # shapes structures (PolyData, Image, etc.) and
            # some losses/morphings are limited to specific shapes
            assert annotations[key] == reference_annotations[
                key
            ] or annotations[key] in get_args(reference_annotations[key])
        else:
            assert issubclass(annotations[key], reference_annotations[key])


def check_module_annotations(module, template, type):
    """Check that the type and the classes' annotations of the methods of a
    module are correct. More precisely, we check that all the classes in the
    module are subclasses of the given type, then we check that the methods of
    the template are in the class, and finally we check that the
    annotations of the methods are correct.

    Args:
        module : the module to check
        template (dict): a dictionary of the form
                            {method_name: {annotation_name: annotation_type}}
        type : the desired type of the classes in the module
    """

    # Find the classes in the module
    for name, cl in inspect.getmembers(module):
        if inspect.isclass(cl):

            # if cl is a subclass of type
            if issubclass(cl, get_args(type)):

                try:
                    # instantiate the class
                    obj = cl()
                except TypeError:
                    raise AssertionError(
                        f"Class {cl} cannot be instantiated with no arguments"
                    )

                print("Inspecting annotations of class {}:".format(name))
                for method_name in template.keys():
                    print("Method name: {}".format(method_name))
                    # Check that the method is in the class
                    assert hasattr(obj, method_name)
                    annotations = getattr(obj, method_name).__annotations__
                    print("Annotations: {}".format(annotations))
                    print(
                        "Reference annotations: {}".format(
                            template[method_name]
                        )
                    )
                    # Check that the annotations are correct
                    check_annotations(
                        annotations=annotations,
                        reference_annotations=template[method_name],
                    )


# Define the templates for the annotations of Loss and Morphing
loss_template = {
    "__call__": {
        "source": sks.shape_type,
        "target": sks.shape_type,
        "return": sks.FloatScalar,
    }
}

morphing_template = {
    "morph": {
        "shape": sks.shape_type,
        "return_path": bool,
        "return_regularization": bool,
        "return": sks.morphing.utils.MorphingOutput,
    }
}


# Define the tests
def test_losses():
    check_module_annotations(
        module=sks.loss,
        template=loss_template,
        type=sks.Loss,
    )


def test_morphing():
    check_module_annotations(
        module=sks.morphing,
        template=morphing_template,
        type=sks.Model,
    )
