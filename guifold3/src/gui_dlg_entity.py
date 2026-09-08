# Copyright 2022 Friedrich Miescher Institute for Biomedical Research
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Georg Kempf, Friedrich Miescher Institute for Biomedical Research

from html.entities import entitydefs
from selectors import EpollSelector
import string
import pkg_resources
from PyQt5 import QtWidgets, uic
import logging
from datetime import date

from guifold3.src.gui_classes import BondEntity, DnaEntity, LigandEntity, ProteinEntity, RnaEntity
from guifold3.src.gui_dialogs import message_dlg

logger = logging.getLogger("guifold3")

class DefaultValuesProtein:
    def __init__(self):
        self.name = None
        self.type = "protein"
        self.copies = 1
        self.sequence = None
        self.use_msa = True
        self.use_template = True
        self.custom_unpaired_msa_path = None
        self.custom_paired_msa_path = None
        self.modification_type = None
        self.modification_position = None

class DefaultValuesDna:
    def __init__(self):
        self.name = None
        self.type = "dna"
        self.copies = 1
        self.sequence = None

class DefaultValuesRna:
    def __init__(self):
        self.name = None
        self.type = "rna"
        self.copies = 1
        self.sequence = None
        self.use_msa = True

class DefaultValuesLigand:
    def __init__(self):
        self.name = None
        self.type = "ligand"

class DefaultValuesBond:
    def __init__(self):
        self.name = None
        self.type = "bond"

class EntityDlg(QtWidgets.QDialog):
    def __init__(self, _parent, entity=None, update=False, entity_type='protein'):
        super(EntityDlg, self).__init__()
        self.update = update
        self._parent = _parent
        self.sess = _parent.sess
        self.db = _parent.db
        self.entity_type = entity_type
        self.error = False
        if self.update:
            self.entity = entity
        else:
            if entity_type == 'protein':
                self.entity = ProteinEntity()
            elif entity_type == 'dna':
                self.entity = DnaEntity()
            elif entity_type == 'rna':
                self.entity = RnaEntity()
            elif entity_type == 'ligand':
                self.entity = LigandEntity()
            elif entity_type == 'bond':
                self.entity = BondEntity()
            self.entity.set_db(self.db)
        self.gui_params = _parent.gui_params
        uic.loadUi(pkg_resources.resource_filename('guifold3.ui', f'add_{entity_type}.ui'), self)
        self.entity.set_controls(self, self.entity.db_table)
        self.button_box = self.findChild(QtWidgets.QDialogButtonBox, f"btn_{entity_type}entity_button_box")
        if entity_type in ['protein', 'dna', 'rna']:
            self.btn_modification_add = self.findChild(QtWidgets.QPushButton, f"btn_{entity_type}entity_modification_add")
            self.btn_modification_remove = self.findChild(QtWidgets.QPushButton, f"btn_{entity_type}entity_modification_remove")
            self.btn_modification_add.clicked.connect(self.OnBtnModificationAdd)
            self.btn_modification_remove.clicked.connect(self.OnBtnModificationRemove)
        #if entity_type == 'bond':
        #    alphabet_upper = list(string.ascii_uppercase)
        #    alphabet_lower = list(string.ascii_lowercase)
            #chain_ids = alphabet_upper + alphabet_lower 
            #for item in chain_ids:
            #    self.entity.entity1.ctrl.addItem(item)
            #    self.entity.entity2.ctrl.addItem(item)
        ok_button = self.button_box.button(QtWidgets.QDialogButtonBox.Ok)
        cancel_button = self.button_box.button(QtWidgets.QDialogButtonBox.Cancel)
        ok_button.clicked.connect(self.OnAccept)
        cancel_button.clicked.connect(self.OnReject)

        self.init()

    def init(self):
        if self.entity_type == 'bond':
            # Fill entity combo box with available entities
            self.entity.entity1.cmb_dict = {}
            self.entity.entity2.cmb_dict = {}
            self.entity.entity1.ctrl.clear()
            self.entity.entity2.ctrl.clear()
            for i, obj in enumerate(self._parent.jobparams.get_entity_list()):
                if not obj.type.get_value() == "bond":
                    self.entity.entity1.cmb_dict[i] = obj.name.get_value()
                    self.entity.entity1.ctrl.addItem(self.entity.entity1.cmb_dict[i])
                    self.entity.entity2.cmb_dict[i] = obj.name.get_value()
                    self.entity.entity2.ctrl.addItem(self.entity.entity2.cmb_dict[i])

        if self.update:
            self.entity.update_from_self()
        else:
            if self.entity_type == 'protein':
                default_values = DefaultValuesProtein()
            elif self.entity_type == 'dna':
                default_values = DefaultValuesDna()
            elif self.entity_type == 'rna':
                default_values = DefaultValuesRna()
            elif self.entity_type == 'ligand':
                default_values = DefaultValuesLigand()
            elif self.entity_type == 'bond':
                default_values = DefaultValuesBond()

            self.entity.update_from_default(default_values)

    def OnBtnModificationAdd(self):
        mod_type = self.entity.modification_type.ctrl.text()
        if len(mod_type) != 3:
            self.error_dlg('Error', "Modification type must be a 3-letter CCD code")
        mod_pos = self.entity.modification_position.ctrl.value()
        self.entity.modifications.ctrl.addItem(f"{mod_type}_{mod_pos}")

    def check_sequence(self, sequence: str):
        return True
    
    def error_dlg(self, title: str, text: str) -> None:
        dlg = QtWidgets.QMessageBox(self)
        dlg.setIcon(QtWidgets.QMessageBox.Information)
        dlg.setText(text)
        dlg.setWindowTitle(title)
        return dlg

    def OnBtnModificationRemove(self):
        index = self.entity.modifications.ctrl.currentIndex()
        if index >= 0:
            self.entity.modifications.ctrl.removeItem(index)

    def OnAccept(self):

        self.entity.update_from_gui()
        if hasattr(self.entity, 'sequence'):
            seq = self.entity.sequence.get_value()
            self.entity.sequence.set_value(seq.replace('\n', ''))
        self.error = False
        if not self.entity_type == 'bond':
            logger.debug(f"Value of entity name: {self.entity.name.get_value()}")
            if self.entity.name.get_value() is None and not self.error:
                self.dlg = self.error_dlg("Error", "Name required")
                self.dlg.show()
                self.error = True
            if self.entity.name.get_value() in self._parent.gui_params['current_job_name_list'] and not self.error and not self.update:
                self.dlg = self.error_dlg("Error", "Entity with the same name already exists. Choose a different name.")
                self.dlg.show()
                self.error = True
            if '_' in self.entity.name.get_value() and not self.error:
                self.dlg = self.error_dlg("Error", "_ found in the entity name. This is a reserved character, please choose a different character.")
                self.dlg.show()
                self.error = True
        if hasattr(self.entity, 'sequence'):
            if not self.entity.sequence.get_value() and not self.error:
                self.dlg = self.error_dlg("Error", "Sequence required")
                self.dlg.show()
                self.error = True
            if not self.check_sequence(self.entity.sequence.get_value()) and not self.error:
                self.dlg = self.error_dlg("Error", "Sequence contains forbidden characters. Make sure all characters are standard IUPAC letters for the molecule type")
                self.dlg.show()
                self.error = True
        if self.entity_type == "ligand":
            ligand_info = [self.entity.smiles.get_value(), self.entity.ccd_codes.get_value(), self.entity.cif_file.get_value()]
            if ligand_info.count(None) == 3 and not self.error:
                self.dlg = self.error_dlg("error", "One of smiles, component code or cif file path needs to be supplied.")
                self.dlg.show()
                self.error = True
            elif self.entity.cif_file.get_value() and not self.entity.ccd_codes.get_value() and not self.error:
                self.dlg = self.error_dlg("error", "If a cif file is supplied the component code (CCD) needs to be given as well.")
                self.dlg.show()
                self.error = True
            elif self.entity.smiles.get_value() and any([self.entity.cif_file.get_value(), self.entity.ccd_codes.get_value()]):
                self.dlg = self.error_dlg("error", "If smiles strings are supplied no CCD or cif file can be given at the same time.")
                self.dlg.show()
                self.error = True
        if self.entity_type == "bond":
            if not all([self.entity.entity1.get_value(), self.entity.resid1.get_value(), self.entity.atom1.get_value(),
                       self.entity.entity2.get_value(), self.entity.resid2.get_value(), self.entity.atom2.get_value()]) and not self.error:
                self.dlg = self.error_dlg("Error", "All values have to be set")
                self.dlg.show()
                self.error = True
            else:
                self.entity.name.set_value(f"{self.entity.entity1.get_value()}{self.entity.resid1.get_value()}{self.entity.atom1.get_value()}_{self.entity.entity2.get_value()}{self.entity.resid2.get_value()}{self.entity.atom2.get_value()}")
        params = self.entity.get_dict_db_insert()
        
        if not self.update and not self.error:
            self._parent.gui_params['entity_counter'] += 1
            self.entity.entity_id.set_value(self._parent.gui_params['entity_counter'])
            self.entity.type.set_value(self.entity_type)
            self._parent.add_entity(self.entity)
        if not self.error:            
            super().accept()
        else:
            logger.debug("Error in input")


    def OnReject(self):
        super().reject()