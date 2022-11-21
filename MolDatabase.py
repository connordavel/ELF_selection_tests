# the selected molecules for all tests completed in this folder
from functools import partial
import sys, getopt, os
from pathlib import Path
from tempfile import TemporaryDirectory
from openeye.oechem import All
from rdkit.Chem import AllChem
from openff.toolkit.topology.molecule import FrozenMolecule, Molecule
from openff.toolkit.utils.toolkits import RDKitToolkitWrapper, OpenEyeToolkitWrapper
from openff.toolkit.utils import UndefinedStereochemistryError
from simtk import unit
import pandas as pd
import numpy as np
import subprocess
from distutils.spawn import find_executable
from openff.toolkit.utils.utils import temporary_cd 
from copy import deepcopy
from storage_structure import *

class MolDatabase:
    def __init__(self, files="ALL"):
        self.status = True # status of a the class (are we good to get molecules?)
        stored_files = {
                        "MiniDrugBank.sdf": ("sdf_file", Path( "/home/coda3831/openff-workspace/openforcefield/openff/toolkit/data/molecules/MiniDrugBank.sdf")),
                        # "burn-in.sdf": ("file", "sdf", Path( Path.cwd() / "burn-in.sdf")),
                        "amber-ff-porting": ("amber_proteins", Path("/home/coda3831/openff-workspace/amber-ff-porting/parameter_deduplication/amber-ff-porting"))
        }
        if files=="ALL":
            self.files = stored_files
        elif isinstance(files, list):
            # read only those file that are requested
            self.files = {}
            for file in files:
                if file not in stored_files.keys():
                    print("file not found. Check you file/folder names and try again")
                    self.status = False
                else:
                    self.files[file] = stored_files[file]

    def failed(self):
        off_mol = Molecule()
        off_mol.name = "failed"
        return off_mol, False

    def fix_stereochem(self, molecule: Molecule) -> Molecule:
        try:
            Molecule.from_smiles(molecule.to_smiles())
        except UndefinedStereochemistryError:
            molecule = [molecule, *molecule.enumerate_stereoisomers()][-1]
        return molecule

    def amber_protein_mol_supplier(self, file_path):
        def fix_carboxylate_bond_orders(offmol):
            # function provided by Jeffrey Wagner
            """Fix problem where leap-produced mol2 files have carboxylates defined with all single bonds"""
            # First, find carbanions
            for atom1 in offmol.atoms:
                if atom1.atomic_number == 6 and atom1.formal_charge.value_in_unit(unit.elementary_charge) == -1:
                    # Then, see if they're bound to TWO oxyanions
                    oxyanion_seen = False
                    for bond in atom1.bonds:
                        atom2 = [atom for atom in bond.atoms if not atom == atom1][0]
                        if atom2.element.atomic_number == 8 and atom2.formal_charge.value_in_unit(unit.elementary_charge) == -1:
                            # If we find a bond to a SECOND oxyanion, then zero both 
                            # the carbon and the second oxygen's formal charges, and 
                            # set the bond order to 2
                            if oxyanion_seen:
                                atom1._formal_charge = 0 * unit.elementary_charge
                                atom2._formal_charge = 0 * unit.elementary_charge
                                bond._bond_order = 2
                            oxyanion_seen = True
       
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        if not file_path.exists():
            print("directory does not exists")
            yield self.failed()
        # look for the first combination of the following protein groups
        file_names = []
        pos_proteins = ['LYS', 'HIP', 'ARG']
        neg_proteins = ['GLU', 'ASP']
        for pos in pos_proteins:
            for neg in neg_proteins:
                file_names.append(f"{pos}_{neg}.mol2")
                file_names.append(f"{neg}_{pos}.mol2")
        for file in file_path.glob("**/*.mol2"):
            if file.name in file_names:
                try:
                    off_mol = Molecule.from_file(str(file), file_format="MOL2", allow_undefined_stereo=True)
                    off_mol = self.fix_stereochem(off_mol)
                    if off_mol.n_atoms == 0:
                        yield self.failed()
                    off_mol.name = str(file.parents[1]).split("/")[-1]
                    off_mol.name += "_"
                    off_mol.name += file.stem
                    fix_carboxylate_bond_orders(off_mol)
                    yield off_mol, True
                except Exception:
                    print(f"failed to read protein {file.name}")
                    yield self.failed()
            else:
                continue

    def sdf_mol_file_supplier(self, file_path):
        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        if not file_path.exists():
            print("file does not exists")
            yield self.failed()

        OFF_Mols = Molecule.from_file(str(file_path), 
                                    file_format='sdf', 
                                    toolkit_registry=OpenEyeToolkitWrapper(),
                                    allow_undefined_stereo=True)
        for off_mol in OFF_Mols:
            off_mol = self.fix_stereochem(off_mol)
            yield off_mol, True

    def __iter__(self):
        if self.status == False:
            print("unable to yield values due to unresolved error")
            return
        methods = {"sdf_file": self.sdf_mol_file_supplier,
                   "amber_proteins": self.amber_protein_mol_supplier}
        for name, file in self.files.items():
            type, file_path = file
            method = methods[type]
            for mol in method(file_path):
                yield mol

    # usefull functions 
    def generate_conformers_rdkit_toolkitwrapper(
        self, molecule, n_conformers=1, rms_cutoff=None, clear_existing=True, _cls=None
    ):
        """
        Generate molecule conformers using RDKit.

        .. warning :: This API is experimental and subject to change.

        .. todo ::

           * which parameters should we expose? (or can we implement a general system with \*\*kwargs?)
           * will the coordinates be returned in the OpenFF Molecule's own indexing system? Or is there a chance that they'll get reindexed when we convert the input into an RDMol?

        Parameters
        ----------
        molecule : a :class:`Molecule`
            The molecule to generate conformers for.
        n_conformers : int, default=1
            Maximum number of conformers to generate.
        rms_cutoff : simtk.Quantity-wrapped float, in units of distance, optional, default=None
            The minimum RMS value at which two conformers are considered redundant and one is deleted.
            If None, the cutoff is set to 1 Angstrom

        clear_existing : bool, default=True
            Whether to overwrite existing conformers for the molecule.
        _cls : class
            Molecule constructor

        """
        from rdkit.Chem import AllChem
        rdkit_toolkit_wrapper = RDKitToolkitWrapper()

        if rms_cutoff is None:
            rms_cutoff = 1.0 * unit.angstrom
        rdmol = rdkit_toolkit_wrapper.to_rdkit(molecule)
        # TODO: This generates way more conformations than omega, given the same nConfs and RMS threshold. Is there some way to set an energy cutoff as well?
        AllChem.EmbedMultipleConfs(
            rdmol,
            numConfs=n_conformers,
            pruneRmsThresh=rms_cutoff / unit.angstrom,
            randomSeed=1,
            numThreads=4,
            # params=AllChem.ETKDG()
        )
        molecule2 = rdkit_toolkit_wrapper.from_rdkit(
            rdmol, allow_undefined_stereo=True, _cls=molecule.__class__
        )

        if clear_existing:
            molecule._conformers = list()

        for conformer in molecule2._conformers:
            molecule._add_conformer(conformer)


    def generate_confs(self, mol, method="openeye", n_conformers=200, rms_cutoff= 0.0 * unit.angstrom):
        conformers_generated = True
        if method=="rdkit":
            try:
                # mol.generate_conformers(n_conformers=n_conformers, toolkit_registry = RDKitToolkitWrapper())
                self.generate_conformers_rdkit_toolkitwrapper(mol, n_conformers=n_conformers, rms_cutoff=rms_cutoff)
            except Exception as e:
                print(f"\tRDKit also could not generate conformers for {mol.name}. Continuing with single conformer")
                conformers_generated = False
        elif method=="openeye":
            try:
                mol.generate_conformers(n_conformers=n_conformers, toolkit_registry = OpenEyeToolkitWrapper())
            except Exception:
                print(f"\tOpenEye also could not generate conformers for {mol.name}. Continuing with single conformer")
                conformers_generated = False
        if len(mol.conformers) == 0:
            print(f"no new conformers generated for {mol.name}")
            conformers_generated = False
        return conformers_generated

    def _antechamber_charges(self, mol, apply_BCCs = True):
        def simulate_conformer(conformer, arguments):
            # create the output directory if it doesn't already exist
            # net_charge = mol_copy.total_charge / unit.elementary_charge
            mol_copy._conformers = [conformer]
            # Write out molecule in SDF format
            OpenEyeToolkitWrapper().to_file(
                mol_copy, file_path=f"input_{mol.name}.sdf", file_format="sdf"
            )
            inputs = [
                    "antechamber",
                    "-i",
                    f"input_{mol.name}.sdf",
                    "-fi",
                    "sdf",
                    "-o",
                    f"output_{mol.name}.mol2",
                    "-fo",
                    "mol2",
                    "-pf",
                    "no",
                    "-dr",
                    "n",
                    "-c",
                    short_charge_method,
                    "-nc",
                    str(net_charge),
                ]
            for arg in arguments:
                inputs.append(arg)

            try:
                subprocess.check_output(inputs)

                # output charges for later comparison
                subprocess.check_output(
                    [
                        "antechamber",
                        "-dr",
                        "n",
                        "-i",
                        f"output_{mol.name}.mol2",
                        "-fi",
                        "mol2",
                        "-o",
                        f"output_{mol.name}.mol2",
                        "-fo",
                        "mol2",
                        "-c",
                        "wc",
                        "-cf",
                        f"charges_{mol.name}.txt",
                        "-pf",
                        "no",
                    ]
                )
            except Exception:
                return False
            with open(f"charges_{mol.name}.txt", "r") as infile:
                contents = infile.read()
            text_charges = contents.split()
            charges = np.zeros([mol_copy.n_atoms], np.float64)
            for index, token in enumerate(text_charges):
                charges[index] = float(token)
            charges = unit.Quantity(charges, unit.elementary_charge)

            return charges

        if not isinstance(mol, FrozenMolecule):
            raise TypeError(
                "\"mol\" argument must of class openff.toolkit.topology.Molecule \
                OpenEye and RDkit molecules are not accepted"
            )
        if apply_BCCs:
            short_charge_method = "bcc"
        else:
            short_charge_method = "mul"
        ANTECHAMBER_PATH = find_executable("antechamber")
        if ANTECHAMBER_PATH is None:
            raise FileNotFoundError(
                "Antechamber not found, cannot run "
                "AmberToolsToolkitWrapper.assign_fractional_bond_orders()"
            )
        # create copy, In future, can add conformers functionality like in the main toolkit
        mol_copy = Molecule(mol)
        net_charge = int(mol_copy.total_charge / unit.elementary_charge)

        with TemporaryDirectory() as temp_op_dir:
            with temporary_cd(temp_op_dir):
                args = ["-ek", 
                    f"qm_theory='AM1', grms_tol=0.0005, scfconv=1.d-10, ndiis_attempts=700, qmcharge={net_charge}, maxcyc=0"]
                charges = simulate_conformer(mol_copy.conformers[0], args)

        return charges
    def _openeye_charges(
        self,
        molecule,
        partial_charge_method=None,
        use_conformers=None,
        strict_n_conformers=False,
        _cls=None,):
        """
        Compute partial charges with OpenEye quacpac, and assign
        the new values to the partial_charges attribute.

        .. warning :: This API is experimental and subject to change.

        .. todo ::

           * Should the default be ELF?
           * Can we expose more charge models?


        Parameters
        ----------
        molecule : openff.toolkit.topology.Molecule
            Molecule for which partial charges are to be computed
        partial_charge_method : str, optional, default=None
            The charge model to use. One of ['amberff94', 'mmff', 'mmff94', `am1-mulliken`, 'am1bcc',
            'am1bccnosymspt', 'am1bccelf10']
            If None, 'am1-mulliken' will be used.
        use_conformers : iterable of simtk.unit.Quantity-wrapped numpy arrays, each with shape (n_atoms, 3) and dimension of distance. Optional, default = None
            Coordinates to use for partial charge calculation. If None, an appropriate number of conformers will be generated.
        strict_n_conformers : bool, default=False
            Whether to raise an exception if an invalid number of conformers is provided for the given charge method.
            If this is False and an invalid number of conformers is found, a warning will be raised.
        _cls : class
            Molecule constructor

        Raises
        ------
        ChargeMethodUnavailableError if the requested charge method can not be handled by this toolkit

        ChargeCalculationError if the charge method is supported by this toolkit, but fails
        """

        import numpy as np
        from openeye import oechem, oequacpac

        from openff.toolkit.topology import Molecule

        SUPPORTED_CHARGE_METHODS = {
            "am1bcc": {
                "oe_charge_method": oequacpac.OEAM1BCCCharges,
                "min_confs": 1,
                "max_confs": 1,
                "rec_confs": 1,
            },
            "am1-mulliken": {
                "oe_charge_method": oequacpac.OEAM1Charges,
                "min_confs": 1,
                "max_confs": 1,
                "rec_confs": 1,
            },
            "am1-mullikennoopt": {
                "oe_charge_method": oequacpac.OEAM1Charges,
                "min_confs": 1,
                "max_confs": 1,
                "rec_confs": 1,
            },
            "am1-mullikennosymspt": {
                "oe_charge_method": oequacpac.OEAM1Charges,
                "min_confs": 1,
                "max_confs": 1,
                "rec_confs": 1,
            },
            "gasteiger": {
                "oe_charge_method": oequacpac.OEGasteigerCharges,
                "min_confs": 0,
                "max_confs": 0,
                "rec_confs": 0,
            },
            "mmff94": {
                "oe_charge_method": oequacpac.OEMMFF94Charges,
                "min_confs": 0,
                "max_confs": 0,
                "rec_confs": 0,
            },
            "am1bccnosymspt": {
                "oe_charge_method": oequacpac.OEAM1BCCCharges,
                "min_confs": 1,
                "max_confs": 1,
                "rec_confs": 1,
            },
            "am1bccnoopt": {
                "oe_charge_method": oequacpac.OEAM1BCCCharges,
                "min_confs": 1,
                "max_confs": 1,
                "rec_confs": 1,
            },
            "am1elf10": {
                "oe_charge_method": oequacpac.OEELFCharges(
                    oequacpac.OEAM1Charges(optimize=True, symmetrize=True), 10
                ),
                "min_confs": 1,
                "max_confs": None,
                "rec_confs": 500,
            },
            "am1bccelf10": {
                "oe_charge_method": oequacpac.OEAM1BCCELF10Charges,
                "min_confs": 1,
                "max_confs": None,
                "rec_confs": 500,
            },
        }

        if partial_charge_method is None:
            partial_charge_method = "am1-mulliken"

        partial_charge_method = partial_charge_method.lower()

        if partial_charge_method not in SUPPORTED_CHARGE_METHODS:
            print(
                f"partial_charge_method '{partial_charge_method}' is not available from OpenEyeToolkitWrapper. "
                f"Available charge methods are {list(SUPPORTED_CHARGE_METHODS.keys())} "
            )

        charge_method = SUPPORTED_CHARGE_METHODS[partial_charge_method]

        if _cls is None:
            from openff.toolkit.topology.molecule import Molecule

            _cls = Molecule

        # Make a temporary copy of the molecule, since we'll be messing with its conformers
        mol_copy = _cls(molecule)

        if use_conformers is None:
            if charge_method["rec_confs"] == 0:
                mol_copy._conformers = None
            else:
                OpenEyeToolkitWrapper().generate_conformers(
                    mol_copy,
                    n_conformers=charge_method["rec_confs"],
                    rms_cutoff=0.25 * unit.angstrom,
                )
                # TODO: What's a "best practice" RMS cutoff to use here?
        else:
            mol_copy._conformers = None
            for conformer in use_conformers:
                mol_copy._add_conformer(conformer)

        oemol = mol_copy.to_openeye()

        errfs = oechem.oeosstream()
        oechem.OEThrow.SetOutputStream(errfs)
        oechem.OEThrow.Clear()

        # The OpenFF toolkit has always supported a version of AM1BCC with no geometry optimization
        # or symmetry correction. So we include this keyword to provide a special configuration of quacpac
        # if requested.
        if partial_charge_method == "am1bccnosymspt":
            optimize = False
            symmetrize = False
            quacpac_status = oequacpac.OEAssignCharges(
                oemol, charge_method["oe_charge_method"](optimize, symmetrize)
            )
        elif partial_charge_method == "am1bccnoopt":
            optimize = False
            symmetrize = True
            quacpac_status = oequacpac.OEAssignCharges(
                oemol, charge_method["oe_charge_method"](optimize, symmetrize)
            )
        elif partial_charge_method == "am1-mullikennoopt":
            optimize = False
            symmetrize = True
            quacpac_status = oequacpac.OEAssignCharges(
                oemol, charge_method["oe_charge_method"](optimize, symmetrize)
            )
        elif partial_charge_method == "am1-mullikennosymspt":
            optimize = False
            symmetrize = False
            quacpac_status = oequacpac.OEAssignCharges(
                oemol, charge_method["oe_charge_method"](optimize, symmetrize)
            )
        else:
            oe_charge_method = charge_method["oe_charge_method"]

            if callable(oe_charge_method):
                oe_charge_method = oe_charge_method()

            quacpac_status = oequacpac.OEAssignCharges(oemol, oe_charge_method)

        oechem.OEThrow.SetOutputStream(oechem.oeerr)  # restoring to original state
        # This logic handles errors encountered in #34, which can occur when using ELF10 conformer selection
        if not quacpac_status:

            oe_charge_engine = (
                oequacpac.OEAM1Charges
                if partial_charge_method == "am1elf10"
                else oequacpac.OEAM1BCCCharges
            )

            if "SelectElfPop: issue with removing trans COOH conformers" in (
                errfs.str().decode("UTF-8")
            ):
                print(
                    f"Warning: charge assignment involving ELF10 conformer selection failed due to a known bug (toolkit issue "
                    f"#346). Downgrading to {oe_charge_engine.__name__} charge assignment for this molecule. More information"
                    f"is available at https://github.com/openforcefield/openff-toolkit/issues/346"
                )
                quacpac_status = oequacpac.OEAssignCharges(oemol, oe_charge_engine())

        if quacpac_status is False:
            print(
                f'Unable to assign charges: {errfs.str().decode("UTF-8")}'
            )

        # Extract and return charges
        ## TODO: Make sure atom mapping remains constant

        charges = unit.Quantity(
            np.zeros(shape=oemol.NumAtoms(), dtype=np.float64), unit.elementary_charge
        )
        for oeatom in oemol.GetAtoms():
            index = oeatom.GetIdx()
            charge = oeatom.GetPartialCharge()
            charge = charge * unit.elementary_charge
            charges[index] = charge

        return charges

    def assign_mol_partial_charges(self, mol, method, inplace=False):
        # expects a molecule with ONLY one conformer
        if len(mol.conformers) != 1:
            print("input expects only one conformer. Returning")
            return []
        elif method == "openeyebcc":
            mol_copy = Molecule(mol)
            charges = self._openeye_charges(mol_copy, partial_charge_method="am1bccnoopt", use_conformers=[mol_copy.conformers[0]])
        elif method == "antechamberbcc":
            mol_copy = Molecule(mol)
            charges = self._antechamber_charges(mol_copy)
        elif method == "openeye-mullikennosymspt":
            mol_copy = Molecule(mol)
            charges = self._openeye_charges(mol_copy, partial_charge_method="am1-mullikennosymspt", use_conformers=[mol_copy.conformers[0]])
        elif method == "ante-am1-mullikennosymspt":
            mol_copy = Molecule(mol)
            charges = self._antechamber_charges(mol_copy, apply_BCCs = False)
        elif method == "openeye-mullikennoopt":
            mol_copy = Molecule(mol)
            charges = self._openeye_charges(mol_copy, partial_charge_method="am1-mullikennoopt", use_conformers=[mol_copy.conformers[0]])
        else:
            print(f"method [{method}] is not supported")
            return []
        if charges == False:
            return []
        if inplace == True:
            mol.partial_charges = charges
        return charges._value

def confs_same(a, b):
    a = a._value
    b = b._value
    diff = a - b
    difference_score = sum(sum(abs(diff)))
    if abs(difference_score) < 1.0e-3:
        return True
    else:
        return False

def conformer_test1():
    open("oe_confs.txt", "w").write("?name?conf_id?oe_selected?rd_selected?rmse?ante_charges?oe_charges\n")
    open("rd_confs.txt", "w").write("?name?conf_id?oe_selected?rd_selected?rmse?ante_charges?oe_charges\n")
    
    for mol, status in MolDatabase(["MiniDrugBank.sdf"]):
        column_names = ["name", "conf_id", "oe_selected", "rd_selected", "rmse", "ante_charges", "oe_charges"]
        oe_df = pd.DataFrame(columns=column_names)
        rd_df = pd.DataFrame(columns=column_names)
        try:
            print("here")
            # RDKit generated conformers
            mols_generated = MolDatabase([]).generate_confs(mol, n_conformers=20, method='rdkit')
            if mols_generated:
                original_mol = deepcopy(mol)
                mol_copy1 = Molecule(mol)
                mol_copy2 = Molecule(mol)
                RDKitToolkitWrapper().apply_elf_conformer_selection(mol_copy1, limit=1)
                rd_selected_conf = mol_copy1.conformers[0]
                OpenEyeToolkitWrapper().apply_elf_conformer_selection(mol_copy2, limit=1)
                oe_selected_conf = mol_copy2.conformers[0]
                # output the charges for each conformer and indicate if that conformer was selected or not
                conformers = original_mol.conformers
                line = ""
                for conf, conf_id in zip(conformers, range(0, len(conformers))):
                    print("here times two")
                    mol_copy = deepcopy(mol)
                    mol_copy._conformers = [conf]
                    if confs_same(conf, rd_selected_conf):
                        rd_selected = True
                    else:
                        rd_selected = False

                    if confs_same(conf, oe_selected_conf):
                        oe_selected = True
                    else:
                        oe_selected = False

                    ante_charges = MolDatabase([]).assign_mol_partial_charges(mol_copy, method="antechamberbcc")
                    oe_charges = MolDatabase([]).assign_mol_partial_charges(mol_copy, method="openeyebcc")
                    
                    combined_mol = deepcopy(mol)
                    combined_mol._conformers = [conf, oe_selected_conf]
                    try:
                        rdkit_type = combined_mol.to_rdkit()
                        # GetBestRMS
                        rmse = AllChem.GetConformerRMS(rdkit_type, 0 ,1)
                    except Exception:
                        rmse = 0.0
                    ante_charges = ante_charges.tolist()
                    oe_charges = oe_charges.tolist()
                    rd_df = rd_df.append({"name": mol.name,
                                "conf_id": conf_id, 
                                "oe_selected": oe_selected, 
                                "rd_selected": rd_selected, 
                                "rmse": rmse,
                                "ante_charges": ante_charges, 
                                "oe_charges": oe_charges}, ignore_index=True)
            else:
                print("some problem with conformer generation")
            # Openeye generated conformers
            mols_generated = MolDatabase([]).generate_confs(mol, n_conformers=20, method='openeye')
            if mols_generated:
                original_mol = deepcopy(mol)
                RDKitToolkitWrapper().apply_elf_conformer_selection(mol, limit=1)
                rd_selected_conf = mol.conformers[0]
                OpenEyeToolkitWrapper().apply_elf_conformer_selection(mol, limit=1)
                oe_selected_conf = mol.conformers[0]
                # output the charges for each conformer and indicate if that conformer was selected or not
                conformers = original_mol.conformers
                line = ""
                for conf, conf_id in zip(conformers, range(0, len(conformers))):
                    mol_copy = deepcopy(mol)
                    mol_copy._conformers = [conf]
                    if confs_same(conf, rd_selected_conf):
                        rd_selected = True
                    else:
                        rd_selected = False

                    if confs_same(conf, oe_selected_conf):
                        oe_selected = True
                    else:
                        oe_selected = False

                    ante_charges = MolDatabase([]).assign_mol_partial_charges(mol_copy, method="antechamberbcc")
                    oe_charges = MolDatabase([]).assign_mol_partial_charges(mol_copy, method="openeyebcc")

                    combined_mol = deepcopy(mol)
                    combined_mol._conformers = [conf, rd_selected_conf]
                    try:
                        rdkit_type = combined_mol.to_rdkit()
                        rmse = AllChem.GetConformerRMS(rdkit_type, 0 ,1)
                    except Exception:
                        rmse = 0.0
                    ante_charges = ante_charges.tolist()
                    oe_charges = oe_charges.tolist()
                    oe_df = oe_df.append({"name": mol.name,
                                "conf_id": conf_id, 
                                "oe_selected": oe_selected, 
                                "rd_selected": rd_selected, 
                                "rmse": rmse,
                                "ante_charges": str(ante_charges), 
                                "oe_charges": str(oe_charges)}, ignore_index=True)
                
            else:
                print("some problem with conformer generation")
        except Exception:
            print("some problem with charge or conformer assignment")
        oe_df.to_csv("oe_confs.txt", mode="a", sep="?", header=None)
        rd_df.to_csv("rd_confs.txt", mode="a", sep="?", header=None)

def populate_database():
    engine = create_engine("sqlite:///ELF_selection_tests/big_database.db")
    Session = sessionmaker(bind=engine) # define a session to work on the engine we just made
    session = Session()

    c = 1
    for mol, status in MolDatabase():
        c += 1
        if c > 102:
            break
        print(mol.name)
        try:
            if already_exists(session, mol.name):
                print(f"molecule already processed under the name: {mol.name}")
                continue
            add_molecule(session, mol)
            # RDKit generated conformers
            mols_generated = MolDatabase([]).generate_confs(mol, n_conformers=200, method='rdkit')
            if mols_generated:
                original_mol = deepcopy(mol)
                mol_copy1 = Molecule(mol)
                mol_copy2 = Molecule(mol)
                RDKitToolkitWrapper().apply_elf_conformer_selection(mol_copy1, limit=1)
                rd_selected_conf = mol_copy1.conformers[0]
                OpenEyeToolkitWrapper().apply_elf_conformer_selection(mol_copy2, limit=1)
                oe_selected_conf = mol_copy2.conformers[0]
                # output the charges for each conformer and indicate if that conformer was selected or not
                conformers = original_mol.conformers
                line = ""
                for conf, conf_id in zip(conformers, range(0, len(conformers))):
                    mol_copy = deepcopy(mol)
                    mol_copy._conformers = [conf]
                    if confs_same(conf, rd_selected_conf):
                        rd_selected = True
                    else:
                        rd_selected = False

                    if confs_same(conf, oe_selected_conf):
                        oe_selected = True
                    else:
                        oe_selected = False

                    ante_charges = MolDatabase([]).assign_mol_partial_charges(mol_copy, method="ante-am1-mullikennosymspt")
                    oe_charges = MolDatabase([]).assign_mol_partial_charges(mol_copy, method="openeye-mullikennosymspt")
                    
                    combined_mol = deepcopy(mol)
                    combined_mol._conformers = [conf, oe_selected_conf]
                    try:
                        rdkit_type = combined_mol.to_rdkit()
                        # GetBestRMS
                        rmse = AllChem.GetConformerRMS(rdkit_type, 0 ,1)
                    except Exception:
                        rmse = 0.0
                    ante_charges = ante_charges.tolist()
                    oe_charges = oe_charges.tolist()
                    # add to the database
                    add_conformer_charges(session, 
                                          RDKitConformers,
                                          mol.name,
                                          conf_id,
                                          conf,
                                          oe_charges,
                                          ante_charges,
                                          oe_selected,
                                          rd_selected)
            else:
                print("some problem with conformer generation")
            # Openeye generated conformers
            mols_generated = MolDatabase([]).generate_confs(mol, n_conformers=200, method='openeye')
            if mols_generated:
                original_mol = deepcopy(mol)
                RDKitToolkitWrapper().apply_elf_conformer_selection(mol, limit=1)
                rd_selected_conf = mol.conformers[0]
                OpenEyeToolkitWrapper().apply_elf_conformer_selection(mol, limit=1)
                oe_selected_conf = mol.conformers[0]
                # output the charges for each conformer and indicate if that conformer was selected or not
                conformers = original_mol.conformers
                line = ""
                for conf, conf_id in zip(conformers, range(0, len(conformers))):
                    mol_copy = deepcopy(mol)
                    mol_copy._conformers = [conf]
                    if confs_same(conf, rd_selected_conf):
                        rd_selected = True
                    else:
                        rd_selected = False

                    if confs_same(conf, oe_selected_conf):
                        oe_selected = True
                    else:
                        oe_selected = False

                    ante_charges = MolDatabase([]).assign_mol_partial_charges(mol_copy, method="ante-am1-mullikennosymspt")
                    oe_charges = MolDatabase([]).assign_mol_partial_charges(mol_copy, method="openeye-mullikennosymspt")

                    combined_mol = deepcopy(mol)
                    combined_mol._conformers = [conf, rd_selected_conf]
                    try:
                        rdkit_type = combined_mol.to_rdkit()
                        rmse = AllChem.GetConformerRMS(rdkit_type, 0 ,1)
                    except Exception:
                        rmse = 0.0
                    ante_charges = ante_charges.tolist()
                    oe_charges = oe_charges.tolist()
                    # add to the database
                    add_conformer_charges(session, 
                                          OpenEyeConformers,
                                          mol.name,
                                          conf_id,
                                          conf,
                                          oe_charges,
                                          ante_charges,
                                          oe_selected,
                                          rd_selected)
                
            else:
                print("some problem with conformer generation")
        except Exception as e:
            print(e)
            print("some problem with charge or conformer assignment")

    print("closing and disposing of database") # always do this, especially if you want to run this script continuously
    session.close()
    engine.dispose()
def create_database():
    engine = create_engine("sqlite:///ELF_selection_tests/big_database.db")
    Base.metadata.create_all(engine)
    engine.dispose()

if __name__ == "__main__":
    # for mol, status in MolDatabase():
    #     oe_mol = deepcopy(mol)
    #     rd_mol = deepcopy(mol)
    #     MolDatabase([]).generate_confs(oe_mol, method='openeye')
    #     MolDatabase([]).generate_confs(rd_mol, method='rdkit')
    #     from rdkit.Chem import AllChem
    #     rdkit_type = rd_mol.to_rdkit()
    #     openeye_type = oe_mol.to_rdkit()
    #     rmse = AllChem.GetConformerRMS(rdkit_type, 1 ,2)
    #     print(rmse)
    #     break

    # create_database()
    populate_database()