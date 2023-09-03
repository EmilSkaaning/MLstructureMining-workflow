import os


def test_diffpy_import():
    """
    Test to check if DiffPy-CMI is installed.
    """
    try:
        from diffpy.structure import loadStructure
        from diffpy.srreal.pdfcalculator import PDFCalculator
    except ImportError:
        assert (
            False
        ), "DiffPy-CMI is not installed. Please install it using the provided instructions."


def test_diffpy_simulation():
    """
    Test to check if DiffPy-CMI can simulate a PDF.
    """
    from diffpy.structure import loadStructure
    from diffpy.srreal.pdfcalculator import PDFCalculator

    # Define the path to the sample CIF file
    dir_path = os.path.dirname(
        os.path.realpath(__file__)
    )  # Gets the directory where the current .py file is located
    sample_cif_path = os.path.join(dir_path, "fixtures", "example_structure.cif")

    try:
        # Create a simple simulation
        structure = loadStructure(sample_cif_path)
        pdf_calc = PDFCalculator()
        r, g = pdf_calc(structure)

        # Assert the lengths of r and g to check if the simulation was successful
        assert len(r) > 0 and len(g) > 0, "Failed to generate PDFs"

    except Exception as e:
        assert False, f"An error occurred when testing DiffPy-CMI simulation: {e}"
