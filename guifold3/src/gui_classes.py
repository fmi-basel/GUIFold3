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

from __future__ import absolute_import
import ast
from copy import deepcopy
import hashlib
from importlib.metadata import entry_points
from io import StringIO
import json
import string
from numpy import true_divide
import pkg_resources
import datetime
import sys
import os
import re
import logging
import socket
from Bio import SeqIO
from PyQt5 import QtWidgets, QtGui, QtCore, QtWebEngineWidgets
from jinja2 import FileSystemLoader, Template, Environment, meta, PackageLoader
from guifold3.src.db_helper import DBHelper
from guifold3.src.gui_dialogs import message_dlg, open_files_and_dirs_dlg
from sqlalchemy.orm.exc import NoResultFound
from guifold3.src.gui_threads import LogThread
import traceback
from datetime import date
from shutil import copyfile, rmtree
import configparser
from subprocess import Popen
import nvidia_smi
import math
import shutil
from Bio.PDB import MMCIFParser
from Bio.PDB.mmcifio import MMCIFIO
from typing import Dict, Optional, Tuple, Union, List
from guifold3.src.gui_dlg_advanced_params import DefaultValues as AdvancedSettingsDefaults
import sqlalchemy.orm
import jax
logger = logging.getLogger('guifold3')
#logger.setLevel(logger.DEBUG)

class WebViewer(QtWebEngineWidgets.QWebEngineView):

    def __init__(self, parent=None):
        super(WebViewer, self).__init__(parent)

    def closeEvent(self, qclose_event):
        """Overwrite QWidget method"""
        # Sets the accept flag of the event object, indicates that the event receiver wants the event.
        qclose_event.accept()
        # Schedules this object for deletion, QObject
        self.page().deleteLater()

class NonSelectableGroupItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None):
        super().__init__(parent)

    def flags(self):
        return super().flags() & QtCore.Qt.QNoItemFlags

    def mousePressEvent(self, event):
        pass  # Ignore mouse press events

ctrl_type_dict = {'dsb': QtWidgets.QDoubleSpinBox,
                  'sbo': QtWidgets.QSpinBox,
                  'lei': QtWidgets.QLineEdit,
                  'tbl': QtWidgets.QTableWidget,
                  'cmb': QtWidgets.QComboBox,
                  'cml': QtWidgets.QComboBox,
                  'pte': QtWidgets.QPlainTextEdit,
                  'chk': QtWidgets.QCheckBox,
                  'web': QtWebEngineWidgets.QWebEngineView,
                  'pro': QtWidgets.QProgressBar,
                  'tre': QtWidgets.QTreeWidget,
                  'lbl': QtWidgets.QLabel}

install_path = os.path.dirname(os.path.realpath(sys.argv[0]))

screening_protocol_names = ['first_vs_all', 'all_vs_all', 'first_n_vs_rest', 'grouped_bait_vs_preys', 'grouped_all_vs_all']

class Variable:
    """
        Base model for GUI controls and DB tables. 

        Arguments:

        var_name (str): the name of the variable.
        type (type or None): the data type of the variable.
        db (bool): a flag indicating if the variable should be saved in the database.
        ctrl_type (str): the type of the GUI control.
        db_primary_key (bool): a flag indicating if the variable is a primary key.
        db_foreign_key (str): the foreign key value.
        db_relationship (str): the type of the database relationship.
        db_backref (str): the name of the back-reference.
        cmb_dict (Dict): a dictionary representing the values of a combo box.
        cmd (bool): a flag indicating if the variable is a command.
        required (bool): a flag indicating if the variable is required.
    """
    def __init__(
            self,
            var_name: str = None,
            type_: Union[type, None] = None,
            db: bool = True,
            ctrl_type: str = None,
            db_primary_key: bool = False,
            db_foreign_key: str = None,
            db_relationship: str = None,
            db_backref: str = None,
            cmb_dict: Dict = None,
            cmd: bool = False,
            required: bool = False):

        self.var_name = var_name
        self.value = None
        self.type = type_
        self.ctrl_type = ctrl_type
        self.ctrl = None
        self.db = db
        self.db_relationship = db_relationship
        self.db_backref = db_backref
        self.db_foreign_key = db_foreign_key
        self.db_primary_key = db_primary_key
        self.cmb_dict = cmb_dict
        self.cmd = cmd
        self.selected_item = None
        self.required = required

    def get_value(self) -> Union[list, str, int, float, bool, None]:
        """Get the value of the variable."""
        if self.value in ["None", "none", ""]:
            self.value = None
        return self.value

    def set_ctrl_type(self) -> None:
        """Set the control type."""
        try:
            self.ctrl_type = self.ctrl.__class__.__name__
        except Exception:
            logger.warning("Cannot set control {}".format(self.ctrl.__class__.__name__), exc_info=True)

    def set_value(self, value: Union[list, str, int, float, bool, None]) -> None:
        """ Set the value of the variable."""
        self.value = value

    def add_object_as_list_item(self, value: object) -> None:
        if self.value is None:
            self.value = []
        self.value.append(value)

    def set_cmb_by_text(self, value: str) -> None:
        """Set combo box value by text."""
        if self.ctrl_type == 'cmb':
            for k, v in self.cmb_dict.items():
                logger.debug(f"Setting cmb, {value} in dict {v}")
                if value == v:
                    index = self.ctrl.findText(str(v), QtCore.Qt.MatchFixedString)
                    if index >= 0:
                        self.ctrl.setCurrentIndex(index)
                    else:
                        logger.debug("Item not found")
        else:
            logger.debug("Not a cmb box.")

    def set_control(self, result_val: Union[str, float, int, bool], result_var: str) -> None:
        """Set control value based on control type."""
        if result_val == 'None' or result_val is None:
            result_val = ''
        if self.ctrl_type == 'dsb':
            if result_val != '':
                self.ctrl.setValue(float(result_val))
        elif self.ctrl_type == 'lei':
            self.ctrl.setText(str(result_val))
        elif self.ctrl_type == 'pte':
            self.ctrl.setPlainText(str(result_val))
        elif self.ctrl_type == 'sbo':
            if result_val != '':
                self.ctrl.setValue(int(result_val))
        elif self.ctrl_type == 'cmb':
            for k, v in self.cmb_dict.items():
                logger.debug(f"Setting cmb, result_val {result_val} value in dict {v}")
                if result_val == v:
                    index = self.ctrl.findText(str(v), QtCore.Qt.MatchFixedString)
                    if index >= 0:
                        self.ctrl.setCurrentIndex(index)
                    else:
                        logger.debug("Item not found")
        #Add all items from the result to a combo box
        elif self.ctrl_type == 'cml':
            items = result_val.split(',')
            for item in items:
                logger.debug(f"Adding {item} to {result_val}")
                self.ctrl.clear()
                self.ctrl.addItem(item)
        elif self.ctrl_type == 'chk':
            if not isinstance(result_val, bool):
                if result_val.lower() == 'true':
                    result_val = True
                elif result_val.lower() == 'false':
                    result_val = False
                else:
                    raise ValueError("Value of controlbox must either be true or false!")
            self.ctrl.setChecked(result_val)
        else:
            logger.error("Control type {} not found for control {}!".format(self.ctrl_type, result_var))
        logger.debug(f"Setting {result_var} to {result_val}")

    def unset_control(self) -> None:
        """Unset control value."""
        if hasattr(self, 'ctrl_type') and not self.ctrl_type is None:
            self.ctrl = None

    def is_set(self) -> bool:
        """Check if a variable is set."""
        if isinstance(self.value, list):
            if not self.value == []:
                return True
            else:
                return False
        else:
            if not self.value in [None, "None", "none", ""]:
                return True
            else:
                return False

    def list_like_str_not_all_none(self) -> bool:
        """Check if a string, that is list-like, contains at least one non-None value."""
        if self.value:
            lst = self.value.split(',')
            for e in lst:
                if not e in [None, "None", "none", ""]:
                    return True
        return False

    def list_like_str_not_all_false(self) -> bool:
        """Check if a string, that is list-like, contains at least one True value."""
        if self.value:
            lst = self.value.split(',')
            for e in lst:
                if not e in [False, "False", "false"]:
                    return True
        return False

    def update_from_self(self) -> None:
        """Set control value from variable value."""
        if not self.value is None and not self.ctrl is None:
            logger.debug(f"Set value of control {self.var_name} to {self.value}")
            self.set_control(self.value, self.var_name)

    def update_from_db(self, db_result: Union[str, float, int, bool]) -> None:
        """Set/update variable values from DB """
        logger.debug("Update from DB")
        result = db_result

        if not db_result is None and not db_result == []:
            
            try:
                from sqlalchemy import inspect
                mapper = inspect(result)
                
                for column in mapper.attrs:
                    if self.var_name == 'protein_entities' and column.key == 'proteinentity_collection':
                        entity_list = []
                        for entity in result.proteinentity_collection:
                            base = ProteinEntity()
                            base.update_from_db(entity)
                            entity_list.append(base)
                        self.value = entity_list
                    elif self.var_name == 'dna_entities' and column.key == 'dnaentity_collection':
                        entity_list = []
                        for entity in result.dnaentity_collection:
                            base = DnaEntity()
                            base.update_from_db(entity)
                            entity_list.append(base)
                        self.value = entity_list
                    elif self.var_name == 'rna_entities' and column.key == 'rnaentity_collection':
                        entity_list = []
                        for entity in result.rnaentity_collection:
                            base = RnaEntity()
                            base.update_from_db(entity)
                            entity_list.append(base)
                        self.value = entity_list
                    elif self.var_name == 'ligand_entities' and column.key == 'ligandentity_collection':
                        entity_list = []
                        for entity in result.ligandentity_collection:
                            base = LigandEntity()
                            base.update_from_db(entity)
                            entity_list.append(base)
                        self.value = entity_list
                    elif self.var_name == 'bond_entities' and column.key == 'bondentity_collection':
                        entity_list = []
                        for entity in result.bondentity_collection:
                            base = BondEntity()
                            base.update_from_db(entity)
                            entity_list.append(base)
                        self.value = entity_list
            except sqlalchemy.exc.NoInspectionAvailable:
                pass

            for result_var in vars(db_result):                    
                #logger.debug("DB var: {}\nClass var: {}".format(result_var, self.var_name))
                if result_var == self.var_name:
                    result_val = getattr(db_result, result_var)
                    self.value = result_val
                    logger.debug("Variable name: {}\nValue: {}".format(result_var, result_val))
                    if not self.ctrl is None:
                        if not self.ctrl_type in ['tbl', 'tre']:
                            if result_val in [None, "None", "none"]:
                                if self.ctrl_type == 'chk':
                                    result_val = False
                                else:
                                    result_val = ""
                            self.set_control(result_val, result_var)
        else:
            logger.warning("DB result empty. Nothing to update.")

    def update_from_gui(self) -> None:
        """Set/update attribute values (self.value) from GUI controls"""
        if not self.ctrl is None:
            if self.ctrl_type in ['sbo', 'dsb']:
                    self.value = self.ctrl.value()
            elif self.ctrl_type == 'lei':
                self.value = self.ctrl.text()
            elif self.ctrl_type == 'pte':
                self.value = self.ctrl.toPlainText()
            elif self.ctrl_type == 'chk':
                self.value = self.ctrl.isChecked()
            elif self.ctrl_type == 'cmb':  # and not self.selected_item is None:
                self.value = self.ctrl.currentText()
            elif self.ctrl_type == 'cml':
                items = [self.ctrl.itemText(i) for i in range(self.ctrl.count())]
                logger.debug(f"Items in cml: {items}")
                self.value = ','.join(items)
                #if not isinstance(self.value, str):
                #    self.value = str(self.value)
            #if self.value == "":
            #    self.value = None
            logger.debug(f"Value of {self.var_name}: {self.value}")
        else:
            logger.debug(f"ctrl of {self.var_name} is not bound.")


    def reset_ctrl(self) -> None:
        """Clear variable values and GUI controls """
        logger.debug("Resetting GUI controls ")
        if not self.ctrl is None:
            logger.debug(f"resetting {self.var_name}")
            if not self.ctrl_type in ['chk', 'tbl', 'tre']:
                self.ctrl.clear()
                self.value = None

            elif self.ctrl_type == 'chk':  # and not self.selected_item is None:
                self.ctrl.setChecked(False)
                self.value = False

            elif self.ctrl_type == 'tbl':
                while self.ctrl.rowCount() > 0:
                    self.ctrl.removeRow(0)
            elif self.ctrl_type == 'tre':
                # Get the model associated with the tree view
                model = self.ctrl.model()

                # Clear the model by removing all rows
                model.removeRows(0, model.rowCount())
        else:
            logger.debug(f"ctrl of {self.var_name} is not bound.")


class TblCtrlJobs(Variable):
    """GUI list control which shows jobs for project"""
    def __init__(self,
                var_name: str = None,
                type: str = None,
                db: bool = True,
                ctrl_type: str = None,
                db_primary_key: str = False,
                db_foreign_key: str = None) -> None:
        super().__init__(var_name,
                        type,
                        db,
                        ctrl_type,
                        db_primary_key,
                        db_foreign_key)
        self.selected_item = None

    def init_gui(self, gui_params, *args, **kwargs):
        return gui_params


class WebCtrlEvaluation(Variable):
    """GUI list control which shows the results in a webbrowser"""
    def __init__(self,
                var_name: str = None,
                type: str = None,
                db: bool = True,
                ctrl_type: str = None,
                db_primary_key: str = False,
                db_foreign_key: str = None) -> None:
        super().__init__(var_name,
                        type,
                        db,
                        ctrl_type,
                        db_primary_key,
                        db_foreign_key)


class SelectCustomTemplateWidget(QtWidgets.QWidget):
    """Line edit and button to select custom templates"""
    def __init__(self, parent=None):
        super(SelectCustomTemplateWidget, self).__init__(parent)

        hbox = QtWidgets.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        lei_custom_template = QtWidgets.QLineEdit()
        btn_custom_template = QtWidgets.QToolButton()
        hbox.addWidget(lei_custom_template)
        hbox.addWidget(btn_custom_template)
        self.lei_custom_template = lei_custom_template
        self.btn_custom_template = btn_custom_template
        self.setLayout(hbox)

class SelectPrecomputedMsasWidget(QtWidgets.QWidget):
    """Line edit and button to select precomputed MSAs"""
    def __init__(self, parent=None):
        super(SelectPrecomputedMsasWidget, self).__init__(parent)

        hbox = QtWidgets.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        lei_precomputed_msas = QtWidgets.QLineEdit()
        btn_precomputed_msas = QtWidgets.QToolButton()
        hbox.addWidget(lei_precomputed_msas)
        hbox.addWidget(btn_precomputed_msas)
        self.lei_precomputed_msas = lei_precomputed_msas
        self.btn_precomputed_msas = btn_precomputed_msas
        self.setLayout(hbox)

class ModifyEntityWidget(QtWidgets.QWidget):
    """Line edit and button to select precomputed MSAs"""
    def __init__(self, parent=None):
        super(ModifyEntityWidget, self).__init__(parent)
        hbox = QtWidgets.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        self.btn_modify_entity = QtWidgets.QPushButton()
        self.btn_modify_entity.setText('Edit')
        hbox.addWidget(self.btn_modify_entity)
        self.setLayout(hbox)


class DeleteEntityWidget(QtWidgets.QWidget):
    """Line edit and button to select precomputed MSAs"""
    def __init__(self, parent=None):
        super(DeleteEntityWidget, self).__init__(parent)

        hbox = QtWidgets.QHBoxLayout()
        hbox.setContentsMargins(0, 0, 0, 0)
        self.btn_delete_entity = QtWidgets.QPushButton()
        self.btn_delete_entity.setText('Delete')
        hbox.addWidget(self.btn_delete_entity)
        self.setLayout(hbox)


class TblCtrlEntities(Variable):
    """Table view for managing enetered entities. Inherits `Variable` class."""

    def __init__(self,
                var_name: str = None,
                type: str = None,
                db: bool = True, 
                ctrl_type: str = None,
                db_primary_key: bool = False, 
                db_foreign_key: str = None) -> None:
        super().__init__(var_name, type, db, ctrl_type, db_primary_key, db_foreign_key)
        self.entity_template =  {"modify": [],
                                "delete": []}
        self.entity_widgets = {k: [] for k in self.entity_template.keys()}

    def init_gui(self, gui_params, *args, **kwargs):
        
        self.entity_widgets = {k: [] for k in self.entity_template.keys()}
        gui_params['current_job_name_list'] = []
        self.ctrl.clear()
        self.ctrl.setRowCount(0)
        self.ctrl.setColumnCount(0)
        self.ctrl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.ctrl.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.ctrl.setColumnCount(4)
        self.ctrl.setHorizontalHeaderLabels(("Name",
                                             "Type",
                                             "",
                                             ""))
        self.ctrl.setColumnWidth(0, 200)
        self.ctrl.setColumnWidth(1, 300)
        self.ctrl.setColumnWidth(2, 300)
        self.ctrl.setColumnWidth(3, 200)
        self.ctrl.setStyleSheet("item { align: center; padding: 0px; margin: 0px;}")
        return gui_params

class GUIVariables:
    """This class contains common functions shared by GUI associated variables."""
    def set_controls(self, ui: QtWidgets.QMainWindow, db_table: str) -> None:
        """ Set QtWidget references for all `Variable` objects in the given scope."""
        logger.debug("Setting controls")
        for var in vars(self):
            logger.debug(var)
            
            obj = getattr(self, var)
            logger.debug(obj)
            if hasattr(obj, 'ctrl_type') and not obj.ctrl_type is None:
                logger.debug(f"{obj.ctrl_type}_{db_table}_{obj.var_name}")
                #if obj.ctrl is None:
                ctrl = ui.findChild(ctrl_type_dict[obj.ctrl_type],
                                      '{}_{}_{}'.format(obj.ctrl_type, db_table, obj.var_name))
                logger.debug(ctrl)
                if not ctrl is None:
                    obj.ctrl = ctrl
                    logger.debug(obj.ctrl)
                else:
                    logger.debug("ctrl not found in res")

    def unset_controls(self, ui: QtWidgets.QMainWindow, db_table: str) -> None:
        """Remove QtWidget references from all `Variable` objects in the given scope."""
        logger.debug("Unsetting controls")
        for var in vars(self):
            logger.debug(var)
            obj = getattr(self, var)
            logger.debug(obj)
            if hasattr(obj, 'ctrl_type') and not obj.ctrl_type is None:
                obj.ctrl = None

    def delete_controls(self, var_obj: Variable) -> None:
        """Remove QtWidget references from a given `Variable` object in case it exists in the given scope."""
        logger.debug("Deleting controls")
        for var in vars(self):
            logger.debug(var)
            obj = getattr(self, var)
            if hasattr(obj, 'var_name'):
                logger.debug(f"self object {obj.var_name}")
            for var_ in vars(var_obj):
                logger.debug(f"db object {var_}")
                if hasattr(obj, 'ctrl'):
                    if obj.var_name == var_:
                        logger.debug(f"Deleting {obj.var_name}")
                        obj.ctrl = None

    def update_from_self(self) -> None:
        """Update QtWidget value from `value` paramter of `Variable` objects in the given scope."""
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'ctrl') and hasattr(obj, 'value'):
                obj.update_from_self()

    def update_from_gui(self) -> None:
        """Get QtWidget value and update `value` paramter of `Variable` objects in the given scope."""
        logger.debug("Update from GUI")
        for var in vars(self):
            logger.debug(f"Update {var}")
            obj = getattr(self, var)
            logger.debug(obj)
            if hasattr(obj, 'ctrl'):
                obj.update_from_gui()

    def reset_ctrls(self) -> None:
        """Reset all QtWidget controls in the given scope."""
        logger.debug("Reset ctrls")
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'ctrl'):
                obj.reset_ctrl()

    def update_from_db(self, db_result: list, other=None) -> None:
        """Get values from DB and update QtWidgets."""
        logger.debug("Update from DB")
        if not db_result is None and not db_result == []:
            for var in vars(self):
                obj = getattr(self, var)
                if hasattr(obj, 'ctrl'):
                    #Update only if var is in other
                    if not other is None:
                        logger.debug("Updating only variables in \"other\" object")
                        for var_ in vars(other):
                            logger.debug(f"params var {var}, other var {var_}")
                            if var_ == var:
                                obj.update_from_db(db_result)
                    else:
                        obj.update_from_db(db_result)
        else:
            logger.warning("DB result empty. Nothing to update.")

    def update_from_default(self, default_values: object) -> None:
        """Update `value` paramter of `Variable` objects in the given scope to default values."""
        logger.debug("Update from Default")
        if not default_values is None and not default_values == []:
            for var in vars(self):
                obj = getattr(self, var)
                if hasattr(obj, 'ctrl') and obj.var_name in vars(default_values):
                    obj.update_from_db(default_values)
        else:
            logger.warning("Default values empty. Nothing to update.")

    def get_dict_run_job(self) -> dict:
        """Get dict with values from GUI controls and variables in the given scope."""
        job_dict = {}
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'db') and (hasattr(obj, 'ctrl') or hasattr(obj, 'file_type')):
                if not obj.is_set():
                    obj.value = None
                job_dict[obj.var_name] = obj.value
        return job_dict

    def get_dict_cmd(self, foreign_obj: object = None) -> dict:
        """Get dict with values that are required for running a command."""
        cmd_dict = {}
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'cmd'):
                if obj.cmd is True:
                    if hasattr(obj, 'ctrl_type'):
                        if obj.ctrl_type == 'chk':
                            if not isinstance(obj.value, bool):
                                if not obj.value is None:
                                    if obj.value.lower() == 'true':
                                        cmd_dict[obj.var_name] = ""
                            else:
                                if obj.value:
                                    cmd_dict[obj.var_name] = ""
                        else:
                            if obj.is_set():
                                cmd_dict[obj.var_name] = obj.value
                    else:
                        if obj.is_set():
                            cmd_dict[obj.var_name] = obj.value
        return cmd_dict

    def get_dict_db_insert(self, foreign_obj: object = None) -> List[Dict]:
        """ Gather names and values from Variable objects and construct a mapping for DB insertion. """
        insert_dict = {}
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'db'):
                if obj.db is True and obj.db_primary_key is False and obj.db_foreign_key is None and obj.db_relationship is None:
                    
                    #if obj.is_set():
                    if isinstance(obj.value, list):
                        obj.value = ','.join(obj.value)
                    logger.debug(f"dict db insert var {var} value {obj.value} type {type(obj.value)}")
                    insert_dict[obj.var_name] = obj.value
                elif not obj.db_relationship is None:
                    if not foreign_obj is None:
                        insert_dict[obj.var_name] = foreign_obj
        return [insert_dict]


class Evaluation(GUIVariables):
    """ Class for managing job evaluation results. Inherits functions from GUIVariables.
    
    Attributes:
        db (str): Database name.
        db_table (str): Database table name.
        id (Variable): Evaluation ID.
        job_id (Variable): Job ID.
        results_path (Variable): Path to results.
        results (Variable): Webview widget.
        pbar (Variable): Progress bar.

        """
    def __init__(self) -> None:
        self.db = None
        self.db_table = 'evaluation'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.job_id = Variable('job_id', 'int', db_foreign_key='job.id')
        self.results_path = Variable('results_path', 'str')
        self.results = Variable("results", db=False, ctrl_type='web')
        self.pbar = Variable("pbar", db=False, ctrl_type='pro')

    def set_db(self, db: DBHelper) -> None:
        """Set database helper object."""
        self.db = db

    def check_exists(self, job_id: int, sess: sqlalchemy.orm.Session) -> bool:
        """Check if evaluation for given job exists in DB."""
        result = sess.query(self.db.Evaluation).filter_by(job_id=job_id).all()
        if result == [] or result == None:
            return False
        else:
            return True

    def get_dict_db_insert(self, foreign_obj: object = None) -> List[dict]:
        """Gather names and values from Variable objects and construct a mapping for DB insertion."""
        insert_dict = {}
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'db'):
                if obj.db is True and obj.db_primary_key is False and obj.db_foreign_key is None and obj.db_relationship is None:
                    logger.debug(f"dict db insert var {var} value {obj.value}")
                    if obj.is_set():
                        insert_dict[obj.var_name] = obj.value
                elif not obj.db_relationship is None:
                    if not foreign_obj is None:
                        insert_dict[obj.var_name] = foreign_obj
        return [insert_dict]

    def generate_db_object(self, data: list) -> list:
        """Generate DB object from data."""
        if isinstance(data, list):
            return [self.db.Evaluation(**row) for row in data]
        else:
            return [self.db.Evaluation(**data)]

    def get_results_path_by_id(self, job_id: int, sess: sqlalchemy.orm.Session) -> str:
        """Get results path by job ID."""
        logger.debug(f"get result for job id {job_id}")
        result_evaluation = sess.query(self.db.Evaluation).filter_by(job_id=job_id).one()
        result_job = sess.query(self.db.Job).filter_by(id=job_id).one()
        results_path = result_evaluation.results_path
        job_name = result_job.name
        project_id = result_job.project_id
        project_path = sess.query(self.db.Project.path).filter_by(id=project_id).one()
        #Path inside the project dir
        absolute_results_path = os.path.join(project_path, job_name, results_path)
        return absolute_results_path

    def print_page_info(self, ok) -> None:
        """Print page info."""
        self.pbar.ctrl.hide()

    def print_load_started(self) -> None:
        """Print load started."""
        logger.debug('Started loading')
        self.pbar.ctrl.show()

    def print_load_percent(self, percent: Union[str, int]) -> None:
        """Print load percent."""
        self.pbar.ctrl.setValue(int(percent))
        logger.debug(percent)
        if percent == 100:
            self.pbar.ctrl.hide()

    def init_gui(self, gui_params: dict, other: object = None, sess: sqlalchemy.orm.Session = None) -> dict:
        """Update GUI with evaluation results."""
        if not gui_params['job_id'] is None:
            if not gui_params['results_path'] is None:
                #results_path = self.get_results_path_by_id(gui_params['job_id'], sess)
                results_path = os.path.join(gui_params['results_path'], 'results_model_viewer.html')
                logger.debug(f"Results path: {results_path}")

                logger.debug(results_path)
                self.results.ctrl.settings().setAttribute(QtWebEngineWidgets.QWebEngineSettings.WebGLEnabled, False)
                logger.debug(f"WebGL enabled: {self.results.ctrl.settings().testAttribute(QtWebEngineWidgets.QWebEngineSettings.WebGLEnabled)}")
                logger.debug(f"QMLSCENE_DEVICE: {os.environ.get('QMLSCENE_DEVICE', 'not set')}")

                self.results.ctrl.loadStarted.connect(self.print_load_started)
                self.results.ctrl.loadProgress.connect(self.print_load_percent)
                self.results.ctrl.loadFinished.connect(self.print_page_info)
                self.results.ctrl.load(QtCore.QUrl(f'file://{results_path}'))
                #self.results.ctrl.setUrl(QtCore.QUrl(f'file://{results_path}'))
                gui_params['results_html'] = results_path

        return gui_params

class JobParams(GUIVariables):
    """ Class for managing job specific parameters. 
    
    Attributes:
        db (Database): Database object 
        db_table (str): Name of the database table
        id (Variable): ID of the jobparams entry
        job_id (Variable): ID of the job
        job_name (Variable): Name of the job
        output_dir (Variable): Output directory
        sequences (Variable): Sequences
        seq_names (Variable): Sequence names
        json_path (Variable): Path to fasta file (reflects AF argument)
        sequence_params (TblCtrlSequenceParams): Sequence parameters TableView widget
        num_multimer_predictions_per_model (Variable): Number of multimer predictions per model (reflects AF argument)
        queue (Variable): Whether to use queue submission
        model_preset_dict (dict): Dictionary with model presets
        model_preset (Variable): Model preset (reflects AF argument)
        benchmark (Variable): Toggle benchmarking (reflects AF argument)
        random_seed (Variable): Random seed (reflects AF argument)
        max_template_date (Variable): Maximum template date (reflects AF argument)
        precomputed_msas_path (Variable): Path to precomputed MSAs (reflects AF argument)
        force_cpu (Variable): Toggle CPU usage
        num_recycle (Variable): Number of recycling steps (reflects AF argument)
        pipeline_dict (dict): Dictionary with pipeline presets
        pipeline (Variable): Pipeline preset (reflects AF argument)
        prediction_dict (dict): Dictionary with prediction presets
        prediction (Variable): Prediction protocol preset
        num_gpu (Variable): Number of GPUs 
        """
    def __init__(self):
        self.db = None
        self.db_table = 'jobparams'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.job_id = Variable('job_id', 'int', db_foreign_key='job.id')
        self.protein_entities = Variable('protein_entities', None, db_relationship='Proteinentity', db_backref="jobparams")
        self.dna_entities = Variable('dna_entities', None, db_relationship='Dnaentity', db_backref="jobparams")
        self.rna_entities = Variable('rna_entities', None, db_relationship='Rnaentity', db_backref="jobparams")
        self.ligand_entities = Variable('ligand_entities', None, db_relationship='Ligandentity', db_backref="jobparams")
        self.bond_entities = Variable('bond_entities', None, db_relationship='Bondentity', db_backref="jobparams")
        #self.job = Variable('job', None, db_relationship='Job')

        self.job_name = Variable('job_name', 'str', ctrl_type='lei')
        self.output_dir = Variable('output_dir', 'str', cmd=True)
        self.json_path = Variable('json_path', 'str', cmd=True)
        self.chain_id_name_mapping_json_path = Variable('chain_id_name_mapping_json_path', 'str')
        self.entity_table = TblCtrlEntities('entity_table',
                                                      None,
                                                      db=False,
                                                      ctrl_type='tbl')
        self.entity_list = Variable('entity_list', 'lst', db=False)
        self.conformer_max_iterations = Variable('conformer_max_iterations',
                                                           'int', ctrl_type='sbo', cmd=True)
        self.num_diffusion_samples = Variable('num_diffusion_samples',
                                                           'int', ctrl_type='sbo', cmd=True)
        self.model_seeds = Variable('model_seeds', 'str', ctrl_type='lei')
        self.queue = Variable('queue', 'bool', ctrl_type='chk')
        self.num_seeds = Variable('num_seeds', 'str', ctrl_type='sbo', cmd=True)
        self.max_template_date = Variable('max_template_date', 'str', ctrl_type='lei', cmd=True)
        self.precomputed_msas_path = Variable('precomputed_msas_path', 'str', ctrl_type='lei')
        self.force_cpu = Variable('force_cpu', 'bool', ctrl_type='chk')
        self.num_recycles = Variable('num_recycles', 'int', ctrl_type="sbo", cmd=True)
        self.pipeline_dict = {0: 'full',
                              1: 'run_data_pipeline',
                              2: 'run_inference',}
        self.pipeline = Variable('pipeline', 'str', ctrl_type='cmb', cmb_dict=self.pipeline_dict)
        self.num_gpu = Variable('num_gpu', 'int', ctrl_type="sbo", cmd=True)

    def set_db(self, db: DBHelper) -> None:
        """Set database helper object."""
        self.db = db

    def set_json_paths(self, job_path: str, job_name: str, job_type: str, job_project_id: str) -> None:
        """Set fasta path."""
        self.json_path.set_value(os.path.join(job_path, f"{job_name}_{job_type}_{job_project_id}.json"))
        self.chain_id_name_mapping_json_path.set_value(os.path.join(job_path, f'{job_name}_{job_project_id}', f"chain_id_name_mapping.json"))

    def adjust_job_project_id(self, json_file: str, job_project_id: int) -> None:
        with open(json_file) as fh:
            data = json.load(fh)
            logger.info(f"Load {json_file}")

        if '_' in data['name']:
            name_list = data['name'].split('_')
            #Replace job_project_id
            name_list[-1] = str(job_project_id)

            data['name'] = '_'.join(name_list)

            with open(json_file, 'w') as fh:
                json.dump(data, fh)
            logger.info(f"Save{json_file}")

    def write_json(self, name, chain_id_name_mapping_json_path):
        if self.model_seeds.get_value() is None:
            model_seeds = [1]
        else:
            model_seeds = self.model_seeds.get_value()

        base_dict = {"name": name,
                  "dialect": "alphafold3",
                  "version": 3,
                  "modelSeeds": model_seeds,
                  "sequences": []}
        
        #Additional json with chain_id name mapping
        chain_id_name_mapping = {}
        alphabet_upper = list(string.ascii_uppercase)[::-1]
        alphabet_lower = list(string.ascii_lowercase)[::-1]
        chain_ids = alphabet_lower + alphabet_upper
        for entity in self.get_entity_list():
            if entity.type.get_value() == 'protein':
                
                protein_dict = {"protein": {"sequence": entity.sequence.get_value(),
                                        "id": [chain_ids.pop() for _ in range(entity.copies.get_value())]}}
                for chain_id in protein_dict['protein']['id']:
                    chain_id_name_mapping[chain_id] = entity.name.get_value()
                protein_entity_dict = protein_dict["protein"]
                if entity.use_msa.get_value() is False:
                    protein_entity_dict["unpaired_msa"] = ""
                    protein_entity_dict["paired_msa"] = ""
                if entity.use_templates.get_value() is False:
                    protein_entity_dict["templates"] = ""
                if entity.custom_unpaired_msa_path.get_value():
                    protein_entity_dict["unpairedMsaPath"] = entity.custom_unpaired_msa_path.get_value()
                if entity.custom_paired_msa_path.get_value():
                    protein_entity_dict["pairedMsaPath"] = entity.custom_paired_msa_path.get_value()
                if entity.unpaired_msa.get_value():
                    protein_entity_dict["unpairedMsa"] = entity.unpaired_msa.get_value()
                if  entity.paired_msa.get_value():
                    protein_entity_dict["pairedMsa"] = entity.paired_msa.get_value()
                if entity.modifications.get_value():
                    
                    protein_entity_dict["modifications"] = []
                    for modification in entity.modifications.get_value().split(','):
                        modification_type, modification_position = modification.split('_')
                        protein_entity_dict["modifications"].append({'ptmType': modification_type, 'ptmPosition': int(modification_position)})
                base_dict['sequences'].append(protein_dict)
            elif entity.type.get_value() == 'dna':
                dna_dict = {"dna": {"sequence": entity.sequence.get_value(),
                                        "id": [chain_ids.pop() for _ in range(entity.copies.get_value())]}}
                for chain_id in dna_dict['dna']['id']:
                    chain_id_name_mapping[chain_id] = entity.name.get_value()
                dna_entity_dict = dna_dict['dna']
                if entity.modifications.get_value():
                    dna_entity_dict["modifications"] = []
                    for modification in entity.modifications.get_value().split(','):
                        modification_type, modification_position = modification.split('_')
                        dna_entity_dict["modifications"].append({'modificationType': modification_type, 'basePosition': int(modification_position)})
                base_dict['sequences'].append(dna_dict)
            elif entity.type.get_value() == 'rna':
                rna_dict = {"rna": {"sequence": entity.sequence.get_value(),
                                        "id": [chain_ids.pop() for _ in range(entity.copies.get_value())]}}
                for chain_id in rna_dict['rna']['id']:
                    chain_id_name_mapping[chain_id] = entity.name.get_value()
                rna_entity_dict = rna_dict['rna']
                if entity.use_msa.get_value() is False:
                    rna_entity_dict["unpaired_msa"] = ""
                    rna_entity_dict["paired_msa"] = ""
                if entity.custom_unpaired_msa_path.get_value():
                    rna_entity_dict["unpairedMsaPath"] = entity.custom_unpaired_msa_path.get_value()
                if entity.custom_paired_msa_path.get_value():
                    rna_entity_dict["pairedMsaPath"] = entity.custom_paired_msa_path.get_value()
                if entity.modifications.get_value():
                    rna_entity_dict["modifications"] = []
                    for modification in entity.modifications.get_value().split(','):
                        modification_type, modification_position = modification.split('_')
                        rna_entity_dict["modifications"].append({'modificationType': modification_type, 'basePosition': int(modification_position)})
                base_dict['sequences'].append(rna_dict)
            elif entity.type.get_value() == 'ligand':
                if entity.ccd_codes.get_value():
                    ccd_codes = entity.ccd_codes.get_value().split(',')
                    chain_id = [chain_ids.pop() for _ in range(len(ccd_codes))]
                    ligand_dict = {"ligand": {"ccdCodes": ccd_codes, "id": chain_id}}
                    if entity.cif_file.get_value():
                        cif_file =  entity.cif_file.get_value()
                        base_dict["userCCDPath"] = cif_file
                elif entity.smiles.get_value():
                    smiles = rf'{entity.smiles.get_value()}'
                    chain_id = chain_ids.pop()
                    ligand_dict = {"ligand": {"smiles": smiles, "id": chain_id}}

                base_dict["sequences"].append(ligand_dict)
                for chain_id in ligand_dict['ligand']['id']:
                    chain_id_name_mapping[chain_id] = entity.name.get_value()
            
        #Iterate again for bond to make sure all chain ids are set

        for entity in self.get_entity_list():
            if entity.type.get_value() == 'bond':
                if not "bondedAtomPairs" in base_dict:
                    base_dict["bondedAtomPairs"] = []
                chains1 = []
                chains2 = []
                for chain_id, name in chain_id_name_mapping.items():
                    if entity.entity1.get_value() == name:
                        chains1.append(chain_id)
                    if entity.entity2.get_value() == name:
                        chains2.append(chain_id)
                for chain1 in chains1:
                    for chain2 in chains2:
                        bond_definition = [[chain1, int(entity.resid1.get_value()), entity.atom1.get_value()], [chain2, int(entity.resid2.get_value()), entity.atom2.get_value()]]
                        base_dict["bondedAtomPairs"].append(bond_definition)
        with open(self.json_path.get_value(), 'w')  as fh:
            json.dump(base_dict, fh)

        with open(self.chain_id_name_mapping_json_path.get_value(), 'w') as fh:
            json.dump(chain_id_name_mapping, fh)
        with open(chain_id_name_mapping_json_path, 'w') as fh:
            logger.debug(f"Writing chain_id_name json to {chain_id_name_mapping_json_path}")
            json.dump(chain_id_name_mapping, fh)

                

    def db_insert_params(self, sess: sqlalchemy.orm.Session, data: list = None) -> None:
        """Insert params into database."""
        assert isinstance(data, list)
        rows = [self.db.Jobparams(**row) for row in data]
        for row in rows:
            sess.merge(row)
        sess.commit()

    def generate_db_object(self, data: list = None) -> List[sqlalchemy.orm.Query]:
        """Generate database object."""
        assert isinstance(data, list)
        return [self.db.Jobparams(**row) for row in data]

    def get_params_by_job_id(self, job_id, sess) -> sqlalchemy.orm.Query:
        """Get params by job id."""
        logger.debug(f"get result for job id {job_id}")
        print(job_id)
        print(type(job_id))
        if isinstance(job_id, int):
            result = sess.query(self.db.Jobparams).filter_by(job_id=job_id).one()

        elif isinstance(job_id, list):
            result = sess.query(self.db.Jobparams).filter(self.db.Jobparams.id.in_(job_id)).all()
        return result
    
    def get_protein_names(self) -> str:
        seq_names = self.seq_names.get_value()
        protein_names = seq_names.replace(',', '_')
        return protein_names
    
    def get_entity_list(self) -> list:
        entity_list = []
        
        if self.protein_entities.get_value() is not None:
            if len(self.protein_entities.get_value()) > 0:
                entity_list.extend(self.protein_entities.get_value())
        if self.dna_entities.get_value() is not None:
            if len(self.dna_entities.get_value()) > 0:
                entity_list.extend(self.dna_entities.get_value())
        if self.rna_entities.get_value() is not None:
            if len(self.rna_entities.get_value()) > 0:
                entity_list.extend(self.rna_entities.get_value())
        if self.ligand_entities.get_value() is not None:
            if len(self.ligand_entities.get_value()) > 0:
                entity_list.extend(self.ligand_entities.get_value())
        if self.bond_entities.get_value() is not None:
            if len(self.bond_entities.get_value()) > 0:
                entity_list.extend(self.bond_entities.get_value())
        
        entity_list = list(set(entity_list))
        entity_id_list = [True if obj.entity_id.get_value() == 0 else False for obj in entity_list]
        if all(entity_id_list):
            print("Id for ordering missing for entities. Adding new initial ordering ids.")
            for i, entity in enumerate(entity_list):
                entity.entity_id.set_value(i+1)
            
        sorted_entity_list = sorted(entity_list, key=lambda obj: obj.entity_id.get_value())

        return sorted_entity_list

    def process_single_template(self, template: str, msgs: list, output_folder: str, index: int, multichain: bool = False) -> bool:
        """Process single template.
          Make sure that the input file only contains one model and one chain and if the chain doesn't have id 'A' rename it.
            Save the cif with a new 4 letter filename."""
        template_name = os.path.basename(template)
        if not template.endswith('.cif'):
            msgs.append(f"Custom template {template_name} needs to be in cif format. Use https://mmcif.pdbj.org/converter to convert.")
            return False
        else:
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure('structure', template)
            models = list(structure.get_models())
            if len(models) > 1:
                msgs.append(f"More than one model found in {template_name}. Make sure that only one model and chain is present in the file.")
                return False
            chains = [x.get_id() for x in structure[0].get_chains()]
            if not multichain:
                if len(chains) > 1:
                    msgs.append(f"More than one chain found in {template_name}. Make sure only the chain that matches the target is present in the file.")
                    return False
                elif not 'A' in chains:
                    logger.debug("Chain IDs in template")
                    logger.debug(chains)
                    msgs.append(f"The template chain needs to have id 'A'in {template_name}.")
                    return False

            if index < 10:
                out_name = f'cus{index}'
                out_path = os.path.join(output_folder, f'cus{index}.cif')
            elif index < 100:
                out_name = f'cu{index}'
                out_path = os.path.join(output_folder, f'cu{index}.cif')
            elif index < 1000:
                out_name = f'c{index}'
                out_path = os.path.join(output_folder, f'c{index}.cif')
            else:
                msgs.append("Too many templates.")
                return False
            new_lines = []
            with open(template, 'r') as f:
                lines = f.readlines()
            for line in lines:
                if line.startswith("_pdbx_database_status.entry_id"):
                    new_line = re.sub(r"(_pdbx_database_status.entry_id\s+)\S+", rf'\1{out_name}', line)
                    new_lines.append(new_line)
                else:
                    new_lines.append(line)
            revision_date_lines = [
                '_pdbx_audit_revision_history.data_content_type\t"Structure model"\n',
                '_pdbx_audit_revision_history.major_revision\t1\n',
                '_pdbx_audit_revision_history.minor_revision\t0\n',
                '_pdbx_audit_revision_history.revision_date\t1971-01-01\n',
                '#']
            new_lines.extend(revision_date_lines)
            logger.debug(f"{template} saved as {out_path}")
            with open(out_path, 'w') as f:
                for line in new_lines:
                    f.write(line)
            return True
        
    def collect_precomputed_msas(self):
        pcm_path = self.precomputed_msas_path.get_value()
        collected_pcm = []
        for root, folders, files in os.walk(pcm_path):
            for file in files:
                if file.endswith('_data.json'):
                    json_file = os.path.join(root, file)
                    logger.debug(f"Looking for MSAs in {json_file}")
                    try:
                        with open(json_file) as fh:
                            data = json.load(fh)
                    except Exception:
                        logger.info(f"Could not open {json_file}. Skipping.")
                        continue
                    if 'sequences' in data:
                        for entity in data['sequences']:
                            for k, v in entity.items():
                                if k in ['protein', 'rna']:
                                    pcm_dict = {}
                                    if 'sequence' in entity[k]:
                                        pcm_dict['sequence'] = entity[k]['sequence']
                                    else:
                                        continue
                                    if 'unpairedMsa' in entity[k]:
                                        pcm_dict['unpairedMsa'] = entity[k]['unpairedMsa']
                                    else:
                                        continue
                                    if 'pairedMsa' in entity[k]:
                                        pcm_dict['pairedMsa'] = entity[k]['pairedMsa']
                                    if 'templates' in entity[k]:
                                        pcm_dict['templates'] = entity[k]['templates']
                                    collected_pcm.append(pcm_dict)
        return collected_pcm

            
    def process_custom_template_files(self, job_dir: str) -> Tuple[str, list]:
        """Process custom template files."""
        msgs = []
        new_custom_template_list = []
        multichain_template_path = None
        folder_count = 0
        for i, item in enumerate(self.custom_template_list.get_value().split(',')):
            custom_template_folder = os.path.join(job_dir, f"custom_templates_{folder_count}")
            logging.debug(f"Processing custom template: {item}")
            if not item in [None, "None"]:
                if not os.path.exists(custom_template_folder):
                    folder_count += 1
                    os.mkdir(custom_template_folder)
                if os.path.isdir(item):
                    cif_files = [f for f in os.listdir(item) if f.endswith(".cif")]
                    if len(cif_files) == 0:
                        msgs.append(f"No cif files found in {item}")
                    else:
                        for o, file in enumerate(cif_files):
                            template = os.path.join(item, file)
                            status = self.process_single_template(template, msgs, custom_template_folder, o)
                            if status is False:
                                break
                else:
                    template = item
                    self.process_single_template(template, msgs, custom_template_folder, 0)
                logger.debug(f"{i} {template}")
                new_custom_template_list.append(custom_template_folder)
            else:
                new_custom_template_list.append("None")
        
        if not self.multichain_template_path.get_value() in [None, "None"]:
            template = self.multichain_template_path.get_value()
            if os.path.exists(template):
                custom_template_folder = os.path.join(job_dir, f"custom_templates_{folder_count}")
                if not os.path.exists(custom_template_folder):
                    os.mkdir(custom_template_folder)
                self.process_single_template(template, msgs, custom_template_folder, 0, multichain=True)
            else:
                msgs.append(f"{template} does not exist!")
            multichain_template_path = os.path.join(custom_template_folder, f'cus0.cif')


            
        return ','.join(new_custom_template_list), multichain_template_path, msgs

    def get_name_by_job_id(self, job_id: int, sess: sqlalchemy.orm.Session) -> str:
        """Get job name by job id."""
        result = sess.query(self.db.Jobparams.job_name).filter_by(job_id=job_id).one()
        logger.debug(result[0])
        logger.debug(result)
        return result[0]

    def set_max_template_date(self) -> None:
        """Set max template date."""
        #format 2021-11-02
        cur_date = date.today()
        if not self.max_template_date.is_set():
            self.max_template_date.set_value(cur_date)

    def init_gui(self, gui_params: dict, other: object = None, sess: sqlalchemy.orm.Session = None) -> dict:
        """Update GUI with sequence params."""
        return gui_params


class Job(GUIVariables):
    """This class contains variables and functions for for handling jobs. It inherits from GUIVariables."""
    def __init__(self) -> None:
        self.db = None
        self.db_table = 'job'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.project_id = Variable('project_id', 'int', db_foreign_key='project.id')
        self.jobparams = Variable('jobparams', None, db_relationship='Jobparams', db_backref="Job")
        self.evaluation = Variable('evaluation', None, db_relationship='Evaluation', db_backref="Job")
        self.job_project_id = Variable('job_project_id', 'int', db=True)
        self.name = Variable('name', 'str')
        # self.params = Variable('params', 'int', db_relationship='Params')
        # self.name = Variable('name', 'str')
        self.list = TblCtrlJobs('list', None, db=False, ctrl_type='tre')
        self.timestamp = Variable('timestamp', 'str')
        self.log = Variable('log', 'str', db=False, ctrl_type='pte')
        self.log_file = Variable('log_file', 'str', db=True)
        self.job_status_log_file = Variable('job_status_log_file', 'str', db=True, cmd=True)
        self.status = Variable('status', 'str', db=True)
        self.pid = Variable('pid', 'str', db=True)
        self.host = Variable('host', 'str', db=True)
        self.path = Variable('path', 'str', db=True)
        self.type = Variable('type', 'str', db=True)
        self.tree_item_expanded = Variable('tree_item_expanded', 'bool', db=True)
        self.active = Variable('active', 'bool', db=True)

    #Getters

    def get_status(self, job_id: str, sess: sqlalchemy.orm.Session) -> str:
        result = sess.query(self.db.Job).get(job_id)
        return result.status

    def get_job_project_id(self, job_id: int, project_id: int, sess: sqlalchemy.orm.Session) -> Optional[int]:
        result = sess.query(self.db.Job).filter_by(id=job_id, project_id=project_id).first()
        if result is None:
            return None
        else:
            return result.job_project_id

    def get_project_id_by_job_id(self, job_id, sess):
        result = sess.query(self.db.Job).filter_by(id=job_id).first()
        return result.project_id
    
    def get_max_job_project_id(self, project_id: int, sess: sqlalchemy.orm.Session) -> Optional[int]:
        result = sess.query(self.db.Job).filter_by(project_id=project_id).order_by(self.db.Job.job_project_id.desc()).first()
        if result is None:
            return None
        else:
            return result.job_project_id
        
    def get_next_job_project_id(self, project_id: int, sess: sqlalchemy.orm.Session) -> int:
        max_id = self.get_max_job_project_id(project_id, sess)
        if max_id is None:
            logger.debug("No job_project_id found.")
            max_id = 0
        return max_id + 1
    
    def get_job_id_by_job_project_id(self, job_project_id: int, project_id: int, sess: sqlalchemy.orm.Session):
        result = sess.query(self.db.Job).filter_by(project_id=project_id, job_project_id=job_project_id).first()
        return result.id

    def get_pid(self, job_id: int, sess: sqlalchemy.orm.Session):
        result = sess.query(self.db.Job).get(job_id)
        logger.debug(f"Getting PID for job_id {job_id} from DB. PID is {result.pid}")
        return result.pid
    
    def get_queue_job_id(self, log_file: str) -> str:
        queue_job_id = None
        logger.debug(f"Reading from {log_file}")
        with open(log_file, 'r') as f:
            lines = f.readlines()
        regex = re.compile("QUEUE_JOB_ID=(\d+)")
        for l in lines:
            if re.search(regex, l):
                queue_job_id = re.search(regex, l).group(1)
        return queue_job_id
    
    def get_host(self, job_id: int, sess: sqlalchemy.orm.Session):
        result = sess.query(self.db.Job).get(job_id)
        return result.host
    
    def get_tree_item_expanded(self, job_name: str, project_id: int, sess: sqlalchemy.orm.Session) -> bool:
        result = sess.query(self.db.Job).filter_by(name=job_name, project_id=project_id).first()
        return result.tree_item_expanded
    
    def get_jobs_by_project_id(self, project_id: int, sess: sqlalchemy.orm.Session):
        result = sess.query(self.db.Job).filter_by(project_id=project_id)
        return result

    def get_type(self, job_params):
        if job_params['pipeline'] in ['run_inference', 'first_vs_all', 'all_vs_all', 'first_n_vs_rest', 'grouped_bait_vs_preys', 'grouped_all_vs_all']:
            type = "prediction"
        elif job_params['pipeline'] in ['run_data_pipeline', 'batch_msas']:
            type = "features"
        elif job_params['pipeline'] == 'only_relax':
            type = "relax"
        else:
            type = "full"
        return type

    def get_type_by_job_id(self, job_id, sess):
        result = sess.query(self.db.Job).get(job_id)
        return result.type    

    def get_active_job_id(self, sess: sqlalchemy.orm.Session) -> int:
        result = sess.query(self.db.Job).filter_by(active=True).first()
        if result:
            return result.id
        else:
            return None
    
    #Setters

    def set_db(self, db) -> None:
        # Set the database to use for this connection.
        self.db = db

    def set_next_job_project_id(self, project_id: str, sess: sqlalchemy.orm.Session) -> None:
        job_project_id = self.get_next_job_project_id(project_id, sess)
        self.job_project_id.value = job_project_id
        logger.debug(f"job_project_id is {job_project_id}")

    def set_type(self, type):
        self.type.set_value(type)

    def set_job_active(self, job_id: int, sess: sqlalchemy.orm.Session) -> int:
        #Set all jobs inactive first
        active_jobs = sess.query(self.db.Job).filter_by(active=True).all()
        for i, item in enumerate(active_jobs):
            active_jobs[i].active = False
        #Set job active by job_id
        job = sess.query(self.db.Job).get(job_id)
        if job:
            job.active = True
        sess.commit()

    #ToDo: check usage and change to var.set()
    def set_project_id(self, project_id):
        self.project_id.value = project_id        

    #DB Updates

    def update_status(self, status: str, job_id: int, sess: sqlalchemy.orm.Session) -> None:
        result = sess.query(self.db.Job).get(job_id)
        result.status = status
        sess.commit()

    def update_tree_item_expanded(self, job_name: str, project_id: int, tree_item_expanded: bool, sess: sqlalchemy.orm.Session) -> None:
        result = sess.query(self.db.Job).filter_by(name=job_name, project_id=project_id).first()
        result.tree_item_expanded = tree_item_expanded
        sess.commit()

    def update_pid(self, pid, job_id, sess):
        logger.debug(f"Updating pid {pid} for job id {job_id}")
        result = sess.query(self.db.Job).get(job_id)
        result.pid = pid
        sess.commit()

    # def convert_protocol(self, cmd_dict):
    #     conversion = {"FastRelax": "1",
    #                   "Backbone Minimization": "2",
    #                   "Allatom Minimization": "3",
    #                   "Automatic rebuilding": "4",
    #                   "Ramachandran-based rebuilding": "5",
    #                   "Only B-factor refinement": "6"}
    #     cmd_dict['protocol'] = conversion[cmd_dict['protocol']]

    def check_queue_submit_cmd(self, cmd):
        if shutil.which(cmd):
            return True
        else:
            return False

    def prepare_submission_script(self, job_params, cmd, estimated_gpu_mem, split_mem):
        msgs = []
        logger.debug("Preparing submit script")
        logger.debug(f"Estimated GPU mem: {estimated_gpu_mem}")
        if 'split_job' in job_params:
            if job_params['split_job_step'] == 'gpu':
                adjust_json = f"adjust_json {job_params['previous_data_json']} {job_params['inference_json']}\n"
                cmd.insert(0, adjust_json)
        command = ''.join(cmd)
        logfile = os.path.join(job_params['job_path'], job_params["log_file"])
        model_dir = job_params['model_dir']
        db_dir = job_params['db_dir']
        submission_script = f"{job_params['job_path']}/submission_script_{job_params['type']}.run"

        #templates = pkg_resources.resource_filename("guifold", "templates")
        submission_script_template = job_params['submission_script_template_path']
        if submission_script_template:
            if os.path.exists(submission_script_template):
                env = Environment(loader=FileSystemLoader(os.path.dirname(submission_script_template)))
                template = env.get_template(os.path.basename(submission_script_template))
                template_source = env.loader.get_source(env, os.path.basename(submission_script_template))
            else:
                msg = f"Custom submission script not found in {submission_script_template}!"
                msgs.append(msg)
                logger.debug(msg)
                return None, msgs
        else:
            env = Environment(loader=PackageLoader("guifold3", "templates"))
            template = env.get_template('submission_script.j2')
            template_source = env.loader.get_source(env, 'submission_script.j2')
            if not os.path.exists(template.filename):
                msg = f"Global submission script template not found in {template.filename}. A custom submission script template can be defined in settings."
                msgs.append(msg)
                logger.debug(msg)
                return None, msgs
        parsed_content = env.parse(template_source)
        template_vars = meta.find_undeclared_variables(parsed_content)
        logger.debug("Template vars:")
        logger.debug(template_vars)
        to_render = {}

        if 'num_cpu' in template_vars:
            if job_params['pipeline'] == 'run_inference':
                to_render['num_cpu'] = 1
            else:
                to_render['num_cpu'] = job_params['num_cpu']
        if 'num_gpus' in template_vars:
            if 'num_gpu' in job_params:
                to_render['num_gpus'] = job_params['num_gpu']
            else:
                to_render['num_gpus'] = 1
        if 'use_gpu' in template_vars:
            if job_params['pipeline'] in ['run_data_pipeline', 'batch_msas'] or job_params['force_cpu']:
                to_render['use_gpu'] = False
            else:
                to_render['use_gpu'] = True
            logger.debug(f"{to_render['use_gpu']} {job_params['force_cpu']}")
        if 'add_dependency' in template_vars:
            to_render['add_dependency'] = job_params['split_job']
        else:
            if job_params['split_job']:
                msgs.append("Job splitting selected in the GUI"
                            " but the submission template is not configured correctly,"
                            " i.e. missing add_dependency parameter. See documentation.")
        if 'queue_job_id' in template_vars:
            to_render['queue_job_id'] = job_params['queue_pid']
        if 'mem' in template_vars:
            if estimated_gpu_mem < job_params['min_ram']:
                ram = job_params['min_ram']
            else:
                if job_params['force_cpu'] or job_params['pipeline'] in ['run_data_pipeline', 'batch_features']:
                    ram = job_params['min_ram']
                else:
                    ram = estimated_gpu_mem
            ram = job_params['min_ram']
            to_render['mem'] = int(ram)
        if 'gpu_mem' in template_vars:
            to_render['gpu_mem'] = estimated_gpu_mem
        if 'account' in template_vars:
            if not job_params['queue_account'] is None:
                to_render['account'] = job_params['queue_account']
            else:
                msgs.append("Queue account name variable specified in template but no account found."
                            " Please specify account in the Settings dialog or remove the variable from the template.")
        if 'split_mem' in template_vars:
            to_render['split_mem'] = split_mem
        if 'total_sequence_length' in template_vars:
            to_render['total_sequence_length'] = job_params['total_seqlen']

        rendered = template.render(**to_render,
                                   model_dir=model_dir,
                                   db_dir=db_dir,
                                   command=command,
                                   logfile=logfile)

        logger.debug(f"Writing to file {submission_script}")
        with open(submission_script, 'w') as f:
            f.write(rendered)
        return submission_script, msgs

    def calculate_gpu_mem_alphafold(self, total_seq_length: int) -> int:
        """Calculate required gpu memory in GB by sequence length for alphafold"""
        logger.debug(total_seq_length)
        #polynomal fit large dataset
        mem = 0.00000504*total_seq_length**2 - 0.00135796*total_seq_length + 5.55021461
        #exponential fit
        #mem = int(math.ceil(4.8898*math.exp(0.00077181*total_seq_length)))
        logger.debug(f"Calculated memory: {mem}")
        return mem

    #Try to get GPU memory from host. Not available when submitted to queue.
    def get_gpu_mem(self):
        nvidia_smi.nvmlInit()
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        return float(info.total) / 10**9

    def get_max_ram(self):
        mem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        mem_gb = int(mem/(1024.**3))
        return mem_gb

    def prepare_cmd(self, job_params, cmd_dict, split_job_step=None):
        error_msgs, warn_msgs = [], []
        estimated_gpu_mem = None
        logger.debug(cmd_dict)
        logger.debug(cmd_dict)
        job_args = []
        log_file = os.path.join(job_params['job_path'], job_params['log_file'])

        #Get compute capability when on a local machine
        special_gpu_settings = False
        if not job_params['queue']:
            gpu_devices = jax.local_devices(backend='gpu')
            if len(gpu_devices) == 0:
                error_msgs.append("Could not find any GPU")
            
            for device in gpu_devices:
                compute_capability = float(device.compute_capability)
                if 7.0 <= compute_capability < 8.0:
                    special_gpu_settings = True


        #Estimate memory
        estimated_gpu_mem = self.calculate_gpu_mem_alphafold(job_params['total_seqlen'])

        #Generate job argument list
        job_args.extend(['--{} {} \\\n'.format(k, v) if i < len(cmd_dict)-1 else '--{} {}'.format(k, v) for i, (k, v) in enumerate(cmd_dict.items())])
        if special_gpu_settings:
            job_args.append('\\\n--flash_attention_implementation xla')
        job_args.append(' \\\n--force_output_dir')
        if job_params['pipeline'] == 'run_data_pipeline':
            job_args.append(' \\\n--norun_inference')
        if job_params['pipeline'] == 'run_inference':
            job_args.append(' \\\n--norun_data_pipeline')

        logger.debug(job_args)
        
        #job_args = [re.sub(r'\sTrue', '', x) for x in job_args if not x is None if not re.search(r'\sFalse$', x)]


        #gpu_name = None
        #gpu_mem = None
        if not 'max_ram' in job_params:
            job_params['max_ram'] = self.get_max_ram()

        #Decide which GPUs to use if several are available on a cluster
        if any([job_params['force_cpu'],
                    job_params['pipeline'] in ['run_data_pipeline', 'batch_msas']]):
            gpu_mem = None
        else:
            if job_params['queue']:
                gpu_mem = job_params['max_gpu_mem']
            else:
                gpu_mem = self.get_gpu_mem()


        split_mem = None
        if not any([gpu_mem is None,
                    estimated_gpu_mem is None,
                    job_params['force_cpu'],
                    job_params['pipeline'] in ['run_data_pipeline', 'batch_msas']]):
            logger.debug(f"\n\n\n\n\n==============Estimated GPU MEM {estimated_gpu_mem} Max GPU mem {gpu_mem}")
            if estimated_gpu_mem > gpu_mem:
                if estimated_gpu_mem > job_params['max_ram']:
                    error_msgs.append(f"The estimated memory of {estimated_gpu_mem} GB for a total sequence length of {job_params['total_seqlen']}"
                                      f" is larger than the maximum availabe GPU memory ({gpu_mem}) GB"
                                      f" and system RAM ({job_params['max_ram']} GB).")
                else:
                    split_mem = estimated_gpu_mem / gpu_mem
                    warn_msgs.append(f"The estimated memory of {estimated_gpu_mem} GB for a total sequence length of {job_params['total_seqlen']}"
                                f" is larger than the availabe GPU memory ({gpu_mem} GB). Confirm to run the job with "
                                f"unified memory (slow; job can only be cancelled from command line)?")
        else:
            logger.debug(f"Skipping GPU memory check")
            logger.debug([gpu_mem is None,
                          estimated_gpu_mem is None,
                          job_params['force_cpu'],
                          job_params['pipeline'] in ['run_data_pipeline', 'batch_msas']])

        cmd = [f"export DEFAULT_MODEL_DIR={job_params['model_dir']}\nexport DEFAULT_DB_DIR={job_params['db_dir']}\n"]
        if special_gpu_settings:
            cmd += ["\nexport XLA_FLAGS='--xla_disable_hlo_passes=custom-kernel-fusion-rewriter'\n"] 
        if not split_mem is None and not job_params['force_cpu']:
            cmd += [f'\nexport TF_FORCE_UNIFIED_MEMORY=True\nexport XLA_PYTHON_CLIENT_MEM_FRACTION={split_mem}\n']
        #cmd = [f"/bin/bash -c \'echo test > {job_params['log_file']}\'"]
        if job_params['pipeline'] in ['run_data_pipeline'] or job_params['force_cpu']:
            cmd += ['\nexport CUDA_VISIBLE_DEVICES=""\nexport JAX_PLATFORMS=cpu\n']
        bin_path = os.path.join(sys.exec_prefix, 'bin')

        cmd += [f"run_alphafold.py \\\n"]
        cmd += job_args 
        if not job_params['queue']:
            cmd += [f">> {log_file} 2>&1"]
        if not job_params['pipeline'] == 'run_data_pipeline':
            cmd += [f" &&\naf3eval \\\n"]
            cmd += [f"--results_dir {job_params['results_path']} "]
            cmd += [f"--chain_id_name_mapping_path {job_params['chain_id_name_mapping_json_path']} "]
        if not job_params['queue']:
            cmd += [f">> {log_file} 2>&1"]
        logger.debug("Job command\n{}".format(cmd))

        if job_params['queue']:
            queue_submit = job_params['queue_submit']
            submission_script, more_msgs = self.prepare_submission_script(job_params, cmd, estimated_gpu_mem, split_mem)
            error_msgs.extend(more_msgs)
            cmd = [queue_submit, submission_script]
        return cmd, error_msgs, warn_msgs, estimated_gpu_mem

    def insert_evaluation(self, _evaluation: object, job_params: dict, sess: sqlalchemy.orm.session.Session) -> bool:
        if not _evaluation.check_exists(job_params['job_id'], sess):
            job_obj = self.get_job_by_id(job_params['job_id'], sess)
            results_path = os.path.join(job_params['results_path'],
                                                            "results_model_viewer.html")
            
            if os.path.exists(results_path):
                _evaluation.results_path.set_value(results_path)
                _evaluation.job_id.set_value(job_params['job_id'])
                evaluation_dict_db = _evaluation.get_dict_db_insert()
                
                evaluation_obj = _evaluation.generate_db_object(evaluation_dict_db)
                job_obj.evaluation_collection.extend(evaluation_obj)
                
                sess.commit()
                return True

            else:
                return False
        else:
            return True
        
    def get_status_dict(self) -> dict:
        task_status_dict = {
                "num_tasks_finished": 0,
                "db_search_started": False,
                "model_1_started": False,
                "model_2_started": False,
                "model_3_started": False,
                "model_4_started": False,
                "model_5_started": False,
                "evaluation_started": False,
                }
        return task_status_dict

    def get_job_status_from_log(self, params):
        logger.debug(f"Updating job status params")
        exit_code = None
        task_status_dict = self.get_status_dict()
        job_status = None
        exit_code_script = None
        exit_code_queue = None
        #try:
        lines = []
        #logger.debug(params)
        log_file = os.path.join(params['job_path'], params['log_file'])


        if params['log_file_lines']:
            #Only lines added since last update
            lines = params['log_file_lines']
        elif os.path.exists(log_file):
            #All lines read again, reset counter
            task_status_dict['num_tasks_finished'] = 0
            logger.debug(f"Reading from {log_file}")
            with open(log_file, 'r') as f:
                lines = f.readlines()
        else:
            msg = f'Log file {log_file} does not exist'
            params['errors'].append(msg)
            return params

        for line in lines:
            pattern_error = re.compile(r'error')
            pattern_exit_code_queue = re.compile(r'Workflow\sfinished\swith\scode\s(\d+)')
            pattern_started = re.compile(r'Running')
            cancelled_pattern = re.compile(r'CANCELLED')
            pattern_finished =  re.compile(r"Done running")
            if re.search(pattern_error, line):
                job_status = 'error'
                logger.debug(f"Exit code from script found in log file {exit_code_script}")
            if re.search(pattern_exit_code_queue, line):
                exit_code_queue = int(re.search(pattern_exit_code_queue, line).group(1))
                logger.debug(f"Exit code from queue found in log file {exit_code_queue}")
            if re.search(cancelled_pattern, line):
                exit_code = 2
                logger.debug(f"Cancelled pattern found in log file {exit_code_queue}")
            if re.search(pattern_started, line):
                job_status = 'running'
            if re.search(pattern_finished, line):
                job_status = 'finished'
        
        logger.debug(f"exit_code_script {exit_code_script} exit_code_queue {exit_code_queue}")
        if exit_code_queue:
            if exit_code_queue > 0:
                exit_code = 1
        
        if not 'task_status' in params:
            params['task_status'] = {}
        params['task_status'] = task_status_dict
        if not 'status' in params:
            params['status'] = "unknown"
        if not job_status is None:
            params['status'] = job_status
        if not 'exit_code' in params:
            params['exit_code'] = None
        params['exit_code'] = exit_code
        logger.debug(params)
        return params


    def db_insert_job(self, sess: sqlalchemy.orm.session.Session = None, data: list = None) -> None:
        assert isinstance(data, list)
        rows = [self.db.Job(**row) for row in data]
        for row in rows:
            sess.merge(row)
        sess.commit()

    def delete_job(self, job_id: int, sess: sqlalchemy.orm.session.Session) -> None:
        sess.query(self.db.Job).filter_by(id=job_id).delete()
        #Cascading delete not yet working
        sess.query(self.db.Jobparams).filter_by(job_id=job_id).delete()
        sess.query(self.db.Evaluation).filter_by(job_id=job_id).delete()
        sess.commit()

    def delete_job_files(self, job_id: str, path: str, sess: sqlalchemy.orm.session.Session) -> None:
        self.delete_job(job_id, sess)
        rmtree(path)

    def set_timestamp(self) -> None:
        self.timestamp.value = datetime.datetime.now()

    def set_host(self) -> None:
        self.host.value = socket.gethostname()

    def hostname(self) -> str:
        return socket.gethostname()

    def generate_db_object(self, data: list = None) -> list:
        assert isinstance(data, list)
        return [self.db.Job(**row) for row in data]

    def get_job_by_id(self, job_id: int, sess: sqlalchemy.orm.session.Session) -> list:
        result = sess.query(self.db.Job).filter_by(id=job_id).one()
        return result

    def get_jobs(self, sess: sqlalchemy.orm.session.Session) -> sqlalchemy.orm.Query:
        result = sess.query(self.db.Job)
        return result

    def read_log(self, log_file: str) -> List[str]:
        lines = []
        logger.debug(f"Reading from {log_file}")
        with open(log_file, 'r') as log:
            lines = log.readlines()
        return lines

    def update_log(self, log_lines: str = None, log_file: str = None, job_id_active: int = None, job_id_thread: int = None, notebook_page: str = None, append: bool = False) -> None:
        """ Updates the log control """
        logger.debug("Updating log")
        if (job_id_active == job_id_thread) or job_id_thread is None:
            if not append:
                self.log.reset_ctrl()
            if log_file:
                if os.path.exists(log_file):
                    logger.debug(f"Reading from {log_file}")
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                else:
                    logger.error(f"Log file {log_file} does not exist!")
                    lines = []
            elif log_lines:
                logger.debug("Found log lines")
                lines = log_lines
            logger.debug("Appending lines")
            for line in lines:
                self.log.ctrl.appendPlainText(line.strip('\n'))
            #self.log_reader_thread = LogThread(log_file, log_lines)
            #self.log_reader_thread.log_updated.connect(self.append_log_lines)
            #self.log_reader_thread.log_updated.connect(self.update_job_status_params)
            #self.log_reader_thread.start()
        else:
            logger.debug("job ids do not match or LogTab not selected")

    def append_log_lines(self, lines: list):
        logger.debug("Appending line")


    def get_path_by_project_id(self, project_id: int, sess: sqlalchemy.orm.session.Session) -> Union[str, None]:
        result = sess.query(self.db.Project.path).filter_by(id=project_id).one()
        return result[0]


    #Name of jobdir in project folder
    def get_job_dir(self, job_name: str, job_project_id: str) -> str:
        job_dir = f"{job_name.lower()}_{job_project_id}"
        return job_dir

    #Full path to job dir
    def get_job_path(self, project_path: str, job_dir: str) -> str:
        job_path = os.path.join(project_path, job_dir)
        return job_path

    def build_log_file_path(self, job_name: str, type_: str, job_project_id: int) -> str:
        log_file = os.path.join(f"{job_name}_{type_}_{job_project_id}.log")
        return log_file

    #Backward compatibility
    def build_log_file_path_deprec(self, job_name: str, type: str) -> str:
        log_file = os.path.join(f"{job_name}_{type}.log")
        return log_file

    def get_log_file(self, job_id: int, sess: sqlalchemy.orm.session.Session) -> str:
        """Reconstruct the log path in case the project path was changed."""
        result_job = sess.query(self.db.Job).filter_by(id=job_id).first()
        result_jobparams = sess.query(self.db.Jobparams).filter_by(job_id=job_id).first()
        job_name = result_jobparams.job_name
        project_id = result_job.project_id
        type = result_job.type
        job_project_id = result_job.job_project_id
        project_path = self.get_path_by_project_id(project_id, sess)
        #Path inside the project dir
        log_file = self.build_log_file_path(job_name.lower(), type, job_project_id)
        log_path = os.path.join(project_path, job_name.lower(), log_file)
        return log_path
    
    def get_job_status_log_file(self, log_file: str) -> str:
        job_status_log_path = "job_status_" + log_file
        return job_status_log_path

    def get_queue_pid(self, log_file: str, job_id: int, sess: sqlalchemy.orm.session.Session) -> Union[str, None]:
        pid = None
        regex = re.compile(r'QUEUE_JOB_ID=(\d+)')
        logger.debug(f"Reading from {log_file}")
        with open(log_file, 'r') as f:
            content = f.read()
        if re.search(regex, content):
            pid = re.search(regex, content).group(1)
        if not pid is None:
            self.update_pid(pid, job_id, sess)
        logger.debug(f"pid from log file is {pid}.")
        return pid
    
    def get_log_file_by_id(self, job_id: int, sess: sqlalchemy.orm.session.Session):
        result = sess.query(self.db.Job).filter_by(id=job_id).one()
        return result.log_file

    def set_log_file(self, log_file: str) -> None:
        self.log_file.value = log_file

    def set_status(self, status: str) -> None:
        self.status.value = status

    def check_pid(self, pid: int) -> bool:
        try:
            os.kill(int(pid), 0)
            logger.debug("PID exists")
            return True
        except Exception:
            logger.debug("PID not found")
            return False
        
    def get_job_name_by_id(self, job_id, sess):
        result = sess.query(self.db.Jobparams.job_name).filter_by(id=job_id).first()
        return result[0]

    def reconnect_jobs(self, sess: sqlalchemy.orm.session.Session) -> list:
        jobs_running = []
        result = sess.query(self.db.Job).filter((self.db.Job.status=="running") | (self.db.Job.status=="starting") | (self.db.Job.status=="waiting")).all()

        for job in result:
            try:
                jobparams = sess.query(self.db.Jobparams).filter(self.db.Jobparams.id == job.id).one()
                jobs_running.append({'job_id': job.id,
                                    'job_project_id': job.job_project_id,
                                    'queue': jobparams.queue,
                                    'status': job.status,
                                    'job_path': job.path,
                                    'job_name': jobparams.job_name,
                                    'log_file': self.get_log_file(job.id, sess),
                                    'pid': job.pid,
                                    'time_started': job.timestamp})
                logger.debug(f"Reconnect job {job.id}, {job.job_project_id}, {jobparams.job_name}, {job.status}")
            except Exception as e:
                logger.debug("Failed to reconnect job", exc_info=True)
        return jobs_running


    def collect_group_subgroup_items(self, parent_item):
        for i in range(parent_item.childCount()):
            child_item = parent_item.child(i)
            item_text = child_item.text(0)
                
            self.collect_group_subgroup_items(child_item)


    #TreeView Implementation
    def init_gui(self, gui_params: dict, reset: bool = False, other: object = None, sess: sqlalchemy.orm.Session = None) -> dict:
        from PyQt5 import QtWidgets

        logger.debug("=== Init Job list ===")
        # Clear Lists
        if reset:
            logger.debug("Restting Job list")
            self.list.reset_ctrl()
        #self.log.reset_ctrl()
        logger.debug("reset end")

        # Fill job list
        self.list.ctrl.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        header_labels = ["Name", "Status", "ID"]
        self.list.ctrl.setHeaderLabels(header_labels)
        #self.list.ctrl.setHeaderHidden(True)
        self.list.ctrl.setColumnWidth(0, 240)
        self.list.ctrl.setColumnWidth(1, 70)
        self.list.ctrl.setColumnWidth(2, 15)
        project_id = gui_params['project_id']

        if project_id is not None:
            job_groups = {}
            existing_items = {}  # Dictionary to store existing items by job ID

            # Iterate through existing top-level items in the tree widget and store them
            for i in range(self.list.ctrl.topLevelItemCount()):
                group_item = self.list.ctrl.topLevelItem(i)
                job_name = group_item.text(0)
                if not job_name in job_groups:
                    job_groups[job_name] = group_item

                # Store sub-items in the dictionary by job ID
                for j in range(group_item.childCount()):
                    sub_item = group_item.child(j)
                    job_project_id = str(sub_item.text(2))
                    existing_items[job_project_id] = sub_item

            jobs = self.get_jobs_by_project_id(project_id, sess)
            jobparams = other.jobparams.get_params_by_job_id([job.id for job in jobs], sess)
            jobparams = [row.__dict__ for row in jobparams]
            logger.debug("Existing items in job list")
            logger.debug(existing_items)


            # Iterate through the fetched job list
            for i, job in enumerate(jobs):
                jobparams_row = [row for row in jobparams if row['id']==job.id][0]
                job_id = job.id
                job_name = job.name
                job_type = job.type
                #jobparams_result = other.jobparams.get_params_by_job_id(job_id, sess=sess)
                pipeline = jobparams_row['pipeline']


                job_project_id = str(job.job_project_id)

                # Check if job already has an item in the tree
                if job_project_id in existing_items:
                    logger.debug(f"{job_project_id} in existing_items")
                    sub_item = existing_items[job_project_id]
                    current_status = sub_item.text(1)
                    new_status = job.status
                    
                    # Update status if it has changed
                    if current_status != new_status:
                        sub_item.setText(1, new_status)

                    # Remove the job ID from the dictionary since it's been processed
                    del existing_items[job_project_id]
                else:
                    logger.debug(f"{job_project_id} not in existing_items")
                    # Create a new group item if necessary
                    if job_name not in job_groups:
                        group_item = QtWidgets.QTreeWidgetItem(self.list.ctrl)
                        group_item.setFlags(group_item.flags() & ~QtCore.Qt.ItemIsSelectable)
                        group_item.setText(0, job_name)
                        self.list.ctrl.addTopLevelItem(group_item)
                        job_groups[job_name] = group_item

                    group_item = job_groups[job_name]
                    sub_item = QtWidgets.QTreeWidgetItem(group_item, [job_type, "", job_project_id])
                    sub_item.setText(1, job.status)

                    if i == jobs.count() - 1 or self.get_tree_item_expanded(job_name, project_id, sess):
                        if not group_item.isExpanded():
                            group_item.setExpanded(True)
                        else:
                            logger.debug("Is already expanded")

            # Remove any items that no longer have corresponding job objects
            for job_id, sub_item in existing_items.items():
                parent_item = sub_item.parent()
                parent_item.removeChild(sub_item)

                if parent_item.childCount() == 0:
                    index = self.list.ctrl.indexOfTopLevelItem(parent_item)
                    self.list.ctrl.takeTopLevelItem(index)
                    job_name = parent_item.text(0)
                    del job_groups[job_name]
            # Automatically scroll to the bottom
            last_item = self.list.ctrl.topLevelItem(self.list.ctrl.topLevelItemCount() - 1)
            self.list.ctrl.scrollToItem(last_item, QtWidgets.QAbstractItemView.PositionAtBottom)

            #self.list.ctrl.expandAll()
            self.list.ctrl.setColumnCount(3)
            #self.list.ctrl.resizeColumnToContents(0)

        if gui_params['job_id'] is None:
            gui_params['job_id'] = self.get_active_job_id(sess)
            if gui_params['job_id']:
                project_id = self.get_project_id_by_job_id(gui_params['job_id'], sess)
                if gui_params['project_id'] == project_id:
                    gui_params['job_name'] = self.get_job_name_by_id(gui_params['job_id'], sess)
                    gui_params['job_project_id'] = self.get_job_project_id(gui_params['job_id'], gui_params['project_id'], sess)
                    gui_params['job_dir'] = self.get_job_dir(gui_params['job_name'], gui_params['job_project_id'])
                    logger.debug(f"job_id is None, getting last active job from DB: {gui_params['job_id']}")
                else:
                    logger.debug("Currently selected project does not match project from last active job. Not switching to last active job")
        else:
            self.set_job_active(gui_params['job_id'], sess)
            logger.debug(f"Setting {gui_params['job_id']} to active.")

        return gui_params


class Entity(GUIVariables):
    def __init__(self):
        self.entity_type_object_mapping = {'protein': 'Proteinentity',
                                            'rna': 'Rnaentity',
                                              'dna': 'Dnaentity',
                                                'ligand': 'Ligandentity',
                                                  'bond': 'Bondentity'}
        self.db_table = None
        self.db = None
        
    def set_db(self, db: DBHelper) -> None:
        self.db = db
        
    def insert_entity(self, data: list, entity_type: str, sess: sqlalchemy.orm.session.Session) -> None:
        table_object = getattr(self.db, self.entity_type_object_mapping[entity_type])
        assert isinstance(data, list)
        rows = [table_object(**row) for row in data]
        for row in rows:
            sess.merge(row)
        sess.commit()

    def generate_db_object(self, data: list = None) -> List[sqlalchemy.orm.Query]:
        """Generate database object."""
        table_object = getattr(self.db, self.entity_type_object_mapping[self.type.get_value()])
        assert isinstance(data, list)
        return [table_object(**row) for row in data]

    def get_entities_by_jobparams_id(self, jobparams_id: int, sess: sqlalchemy.orm.session.Session) -> object:
        table_object = getattr(self.db, self.entity_type_object_mapping[self.type.get_value()])
        return sess.query(table_object).filter(table_object.jobparams_id == jobparams_id) 
        
    def update_entity(self, data: dict, sess: sqlalchemy.orm.session.Session) -> None:
        """Updates a entity in the database."""
        table_object = getattr(self.db, self.entity_type_object_mapping[self.type.get_value()])
        assert isinstance(data, list)
        result = sess.query(table_object).get(self.id)
        for k, v in data[0].items():
            setattr(result, k, v)
        sess.commit()

    def delete(self, sess: sqlalchemy.orm.session.Session) -> None:
        """Deletes an entity from the database and entity_list"""
        if hasattr(self.db, self.entity_type_object_mapping[self.type.get_value()]):
            table_object = getattr(self.db, self.entity_type_object_mapping[self.type.get_value()])
            sess.query(table_object).filter(table_object.id == self.id).delete()
            sess.commit()

    def init_gui(self, gui_params: dict, *args, **kwargs):
        return gui_params

class ProteinEntity(Entity):
    def __init__(self):
        super().__init__()
        
        self.db_table = 'proteinentity'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.entity_id = Variable('entity_id', 'int')
        self.jobparams_id = Variable('jobparams_id', 'int', db_foreign_key='jobparams.id')
        self.name = Variable('name', 'str', ctrl_type='lei')
        self.type = Variable('type', 'str')
        self.sequence = Variable('sequence', 'str', ctrl_type='pte')
        self.copies = Variable('copies', 'int', ctrl_type='sbo')
        self.use_msa = Variable('use_msa', 'bool', ctrl_type='chk')
        self.use_templates = Variable('use_templates', 'bool', ctrl_type='chk')
        self.custom_unpaired_msa_path = Variable('custom_unpaired_msa_path', 'str', ctrl_type='lei')
        self.custom_paired_msa_path = Variable('custom_paired_msa_path', 'str', ctrl_type='lei')
        self.unpaired_msa = Variable('unpaired_msa', 'str', ctrl_type='pte')
        self.paired_msa = Variable('paired_msa', 'str', ctrl_type='pte')
        self.modifications = Variable('modifications', 'str', ctrl_type='cml')
        self.modification_type = Variable('modification_type', 'str', ctrl_type='lei', db=False)
        self.modification_position = Variable('modification_position', 'int', ctrl_type='sbo', db=False)
        self.jobparams = Variable('jobparams', db_relationship="Jobparams", db_backref="protein_entities")

class DnaEntity(Entity):
    def __init__(self):
        super().__init__()
        self.db_table = 'dnaentity'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.entity_id = Variable('entity_id', 'int')
        self.jobparams_id = Variable('jobparams_id', 'int', db_foreign_key='jobparams.id')
        self.name = Variable('name', 'str', ctrl_type='lei')
        self.type = Variable('type', 'str')
        self.sequence = Variable('sequence', 'str', ctrl_type='pte')
        self.copies = Variable('copies', 'int', ctrl_type='sbo')
        self.modifications = Variable('modifications', 'str', ctrl_type='cml')
        self.modification_type = Variable('modification_type', 'str', ctrl_type='lei', db=False)
        self.modification_position = Variable('modification_position', 'int', ctrl_type='sbo', db=False)
        self.jobparams = Variable('jobparams', db_relationship="JobParams", db_backref="dnaentities")

class RnaEntity(Entity):
    def __init__(self):
        super().__init__()
        self.db_table = 'rnaentity'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.entity_id = Variable('entity_id', 'int')
        self.jobparams_id = Variable('jobparams_id', 'int', db_foreign_key='jobparams.id')
        self.name = Variable('name', 'str', ctrl_type='lei')
        self.type = Variable('type', 'str')
        self.sequence = Variable('sequence', 'str', ctrl_type='pte')
        self.copies = Variable('copies', 'int', ctrl_type='sbo')
        self.use_msa = Variable('use_msa', 'bool', ctrl_type='chk')
        self.custom_unpaired_msa_path = Variable('custom_unpaired_msa_path', 'str', ctrl_type='lei')
        self.custom_paired_msa_path = Variable('custom_paired_msa_path', 'str', ctrl_type='lei')
        self.unpaired_msa = Variable('unpaired_msa', 'str', ctrl_type='pte')
        self.paired_msa = Variable('paired_msa', 'str', ctrl_type='pte')
        self.modifications = Variable('modifications', 'str', ctrl_type='cml')
        self.modification_type = Variable('modification_type', 'str', ctrl_type='lei', db=False)
        self.modification_position = Variable('modification_position', 'int', ctrl_type='sbo', db=False)
        self.jobparams = Variable('jobparams', db_relationship="JobParams", db_backref="rna_entities")

class LigandEntity(Entity):
    def __init__(self):
        super().__init__()
        self.db_table = 'ligandentity'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.entity_id = Variable('entity_id', 'int')
        self.jobparams_id = Variable('jobparams_id', 'int', db_foreign_key='jobparams.id')
        self.name = Variable('name', 'str', ctrl_type='lei')
        self.type = Variable('type', 'str')
        self.smiles = Variable('smiles', 'str', ctrl_type='lei')
        self.ccd_codes = Variable('ccd_codes', 'str', ctrl_type='lei')
        self.cif_file = Variable('cif_file', 'str', ctrl_type='lei')
        self.jobparams = Variable('jobparams', db_relationship="JobParams", db_backref="ligand_entities")

class BondEntity(Entity):
    def __init__(self):
        super().__init__()
        self.db_table = 'bondentity'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.entity_id = Variable('entity_id', 'int')
        self.jobparams_id = Variable('jobparams_id', 'int', db_foreign_key='jobparams.id')
        self.name = Variable('name', 'str')
        self.type = Variable('type', 'str')
        self.entity1 = Variable('entity1', 'str', cmb_dict={}, ctrl_type='cmb')
        self.resid1 = Variable('resid1', 'int', ctrl_type='sbo')
        self.atom1 = Variable('atom1', 'str', ctrl_type='lei')
        self.entity2 = Variable('entity2', 'str', cmb_dict={}, ctrl_type='cmb')
        self.resid2 = Variable('resid2', 'int', ctrl_type='sbo')
        self.atom2 = Variable('atom2', 'str', ctrl_type='lei')
        self.jobparams = Variable('jobparams', db_relationship="JobParams", db_backref="bond_entities")


class Project(GUIVariables):
    """ Class for managing projects. 
    
    Attributes:
        db (object): Database object.
        db_table (str): Name of the database table.
        id (object): Variable object for the project id.
        jobs (object): Variable object for the jobs.
        name (object): Variable object for the project name.
        path (object): Variable object for the project path.
        list (object): Variable object for the project list.
        active (object): Variable object for the project active state."""
    def __init__(self):
        self.db = None
        self.db_table = 'project'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.jobs = Variable('jobs', None, db_relationship='Job')
        self.name = Variable('name', 'str', ctrl_type='lei')
        self.path = Variable('path', 'str', ctrl_type='lei')
        self.list = Variable('list', None, db=False, ctrl_type='cmb')
        self.active = Variable('active', 'bool', ctrl_type=None)

    def set_db(self, db: DBHelper) -> None:
        self.db = db

    def insert_project(self, data: list, sess: sqlalchemy.orm.session.Session) -> None:
        """Inserts a new project into the database."""
        assert isinstance(data, list)
        rows = [self.db.Project(**row) for row in data]
        for row in rows:
            sess.merge(row)
        sess.commit()

    def check_if_name_exists(self, project_name: str, sess: sqlalchemy.orm.session.Session) -> bool:
        """Checks if a project already exists in the database."""
        exists = False
        result = self.get_projects(sess)
        for row in result:
            if project_name == row.name:
                exists = True
        return exists
    
    def check_if_path_exists(self, project_path: str, sess: sqlalchemy.orm.session.Session) -> bool:
        """Checks if a project path already exists in the database."""
        exists = False
        result = self.get_projects(sess)
        for row in result:
            if project_path == row.path:
                exists = True
        return exists

    def delete_project(self, project_id: int, sess: sqlalchemy.orm.session.Session) -> None:
        """Deletes a project from the database."""
        sess.query(self.db.Project).filter(self.db.Project.id == project_id).delete()
        #Cascading delete not yet working
        job_ids = sess.query(self.db.Job.id).filter_by(project_id=project_id).all()
        sess.query(self.db.Job).filter_by(project_id=project_id).delete()
        logger.debug("Job IDs to delete")
        logger.debug(job_ids)
        for job_id in job_ids:
            job_id = job_id[0]
            sess.query(self.db.Jobparams).filter_by(job_id=job_id).delete()
            sess.query(self.db.Evaluation).filter_by(job_id=job_id).delete()
        sess.commit()

    def is_empty(self, sess: sqlalchemy.orm.session.Session) -> bool:
        """Checks if the project table is empty."""
        if sess.query(self.db.Project).all() == []:
            return True
        else:
            return False

    def get_project_by_id(self, project_id: int, sess: sqlalchemy.orm.session.Session) -> object:
        return sess.query(self.db.Project).get(project_id)

    def get_active_project(self, sess: sqlalchemy.orm.session.Session) -> Tuple[Union[str, None], Union[int, None]]:
        """Returns the name and id of the active project."""
        try:
            project_name, project_id = sess.query(self.db.Project.name, self.db.Project.id).filter_by(
                active=True).one()
        except NoResultFound:
            project_name, project_id = None, None
        return project_name, project_id

    def get_projects(self, sess: sqlalchemy.orm.session.Session) -> List[object]:
        result = sess.query(self.db.Project).all()
        return result

    def update_project(self, project_id: int, data: dict, sess: sqlalchemy.orm.session.Session) -> None:
        """Updates a project in the database."""
        assert isinstance(data, list)
        result = sess.query(self.db.Project).get(project_id)
        for k, v in data[0].items():
            setattr(result, k, v)
        sess.commit()

    def change_active_project(self, new_project: str, sess: sqlalchemy.orm.session.Session, new_active_id: int = None) -> int:
        """
        Change ative project in DB
        """
        try:
            last_active = sess.query(self.db.Project).filter_by(active=True).one()
            last_active_id = int(last_active.id)
        except:
            last_active_id = -1

        all = sess.query(self.db.Project).all()

        if new_active_id is None:
            new_active = sess.query(self.db.Project).filter_by(name=new_project).one()
            new_active_id = int(new_active.id)

        if not last_active_id == new_active_id:
            # if not last_active_id == -1:
            #    last_active.active = False
            for row in all:
                row.active = False
            new_active.active = True
        sess.commit()

        #result = sess.query(self.db.Project).all()

        return new_active_id

    def get_path_by_project_id(self, project_id: int, sess: sqlalchemy.orm.session.Session) -> str:
        """Returns the path of a project."""
        result = sess.query(self.db.Project).filter_by(id=project_id).first()
        return result.path

    def init_gui(self, gui_params: dict, other: object = None, sess: Union[sqlalchemy.orm.session.Session, None] = None) -> dict:
        logger.debug("=== Init Projects ===")
        projects = self.get_projects(sess)
        # if projects == []:
        #     base_project = [{'name': 'base', 'path': os.getcwd(), 'active': True}]
        #     self.insert_project(base_project)
        #     projects = self.get_projects()
        # logger.debug("Projects")
        # logger.debug(projects)
        if not projects == []:
            self.list.reset_ctrl()
            for item in projects:
                if self.list.ctrl.findText(item.name) == -1:
                    self.list.ctrl.addItem(item.name)
                else:
                    logger.warning("{} not found.".format(item.name))
            name, id = self.get_active_project(sess)
            if name is None:
                for i, item in enumerate(projects):
                    if i == 0:
                        name = item.name
                        id = item.id
                        break
            logger.debug(f"Active project {name} id {id}")
            #Handle case when active project was deleted
            if not name is None:
                gui_params['project_path'] = self.get_path_by_project_id(id, sess)
                projects = self.get_projects(sess)
                # name = projects[0].name
                # id = projects[0].id
                index = self.list.ctrl.findText(name, QtCore.Qt.MatchFixedString)
                logger.debug(f"currently selected project index {index}")
                if index >= 0:
                    self.list.ctrl.setCurrentIndex(index)
                gui_params['project_id'] = id
            return gui_params
        else:
            gui_params['project_id'] = None
            logger.debug("No project found.")
            return gui_params


class Settings(GUIVariables):
    """ Class for managing general settings."""
    def __init__(self) -> None:
        self.db = None
        self.db_table = 'settings'
        self.id = Variable('id', 'int', db_primary_key=True)
        self.queue_submit = Variable('queue_submit', 'str', ctrl_type='lei')
        self.queue_cancel = Variable('queue_cancel', 'str', ctrl_type='lei')
        self.queue_account = Variable('queue_account', 'str', ctrl_type='lei')
        self.submission_script_template_path = Variable('submission_script_template_path', 'str', ctrl_type='lei')
        self.num_cpu = Variable('num_cpu', 'int', ctrl_type='sbo', cmd=True, db=False)
        self.min_cpus = Variable('min_cpus', 'int', ctrl_type='sbo')
        self.max_cpus = Variable('max_cpus', 'int', ctrl_type='sbo')
        self.max_gpu_mem = Variable('max_gpu_mem', 'int', ctrl_type='sbo')
        self.split_job = Variable('split_job', 'bool', ctrl_type='chk')
        self.min_ram = Variable('min_ram', 'int', ctrl_type='sbo')
        self.max_ram = Variable('max_ram', 'int', ctrl_type='sbo')
        #self.queue_submit_dialog = Variable('queue_submit_dialog', 'bool', ctrl_type='chk')
        self.queue_jobid_regex = Variable('queue_jobid_regex', 'str', ctrl_type='lei')
        self.queue_default = Variable('queue_default', 'bool', ctrl_type='chk')
        self.jackhmmer_binary_path =  Variable('jackhmmer_binary_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.nhmmer_binary_path = Variable('nhmmer_binary_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.hmmalign_binary_path = Variable('hmmalign_binary_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.hmmsearch_binary_path = Variable('hmmsearch_binary_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.hmmbuild_binary_path = Variable('hmmbuild_binary_path', 'str', ctrl_type='lei', cmd=True, required=True)
        self.model_dir = Variable('model_dir', 'str', ctrl_type='lei', cmd=True, required=True)
        self.db_dir = Variable('db_dir', 'str', ctrl_type='lei', cmd=True, required=True)
        self.jax_compilation_cache_dir = Variable('jax_compilation_cache_dir', 'str', ctrl_type='lei', cmd=True)
        self.global_config = Variable('global_config', 'bool', ctrl_type='chk')

    def set_db(self, db):
        self.db = db

    def init_gui(self, gui_params: dict, other: object = None, sess: sqlalchemy.orm.Session = None):
        return gui_params

    def get_from_db(self, sess):
        result = sess.query(self.db.Settings).get(1)
        return result

    def check_required_settings(self):
        unset_setting = False
        for var in vars(self):
            obj = getattr(self, var)
            if hasattr(obj, 'required'):
                if obj.required:
                    if obj.value is None or obj.value == '':
                        unset_setting = True
                        logger.debug("obj.var_name is required but not set.")
        return unset_setting


    def get_slurm_account(self):
        account = None
        try:
            p = Popen("sacctmgr show User $(whoami) -p -n &> /dev/null", shell=True)
            stdout, _ = p.communicate()
            if not stdout is None:
                account_raw = stdout.decode()
                logger.debug(f'Output from sacctmgr: {account_raw}')
                account = account_raw.split("|")[1]
        except Exception as e:
            logger.debug("Could not retrieve SLURM account. You can ignore this if you are not using accounts or a different queueing system")
            logger.debug(traceback.print_exc())
        return account

    def set_slurm_account(self, account, sess):
        logger.debug(f"Updating slurm account {account}")
        result = sess.query(self.db.Settings).get(1)
        result.queue_account = account
        sess.commit()

    def update_from_config(self, custom_path=None):
        msgs = []
        if custom_path is None:
            config_file = pkg_resources.resource_filename('guifold3.config', 'guifold.conf')
            logger.debug("update from global config")
        else:
            config_file = custom_path
            logger.debug("update from custom config")
        logger.debug(config_file)
        if os.path.exists(config_file):
            config = configparser.ConfigParser()
            config.read(config_file)
            config_keys = config.keys()
            for section in ['QUEUE', 'DATABASES', 'BINARIES', 'OTHER']:
                if section in config_keys:
                    for key in config[section]:
                        for var in vars(self):
                            if key == var:
                                obj = getattr(self, var)
                                if config[section][key].lower() == 'true':
                                    obj.value = True
                                elif config[section][key].lower() == 'false':
                                    obj.value = False
                                elif not config[section][key] == '':
                                    obj.value = config[section][key]
                else:
                    error_msg = f"{section} section missing from config file"
                    logger.error(error_msg)
                    msgs.append(error_msg)
        else:
            error_msg = f"{config_file} does not exist!"
            logger.error(error_msg)
            msgs.append(error_msg)

        return msgs

    # def get_alphafold_path(self, sess):
    #     result = self.get_from_db(sess)
    #     return result.alphafold_path


    # def check_executables(self, sess):
    #     messages = []
    #     settings = self.get_from_db(sess)
    #     if not settings is None:
    #         if not settings.alphafold_path == '':
    #             self.path.register('alphafold', settings.alphafold_path)
    #         else:
    #             self.path.register('alphafold', None)
    #     else:
    #         self.path.register('alphafold', None)
    #     try:
    #         logger.debug("check alphafold")
    #         self.path.set_exec('alphafold', 'run_alphafold.py')
    #         logger.debug("check alphafold")
    #     except (SystemExit, FileNotFoundError):
    #         messages.append("Alphafold executables not found. Check path in settings!")
    #     except (SystemExit, FileNotFoundError):
    #         pass



        exec_dict = {}
        #result = {'exists': False, 'executable': False}
        #
        #
        # results = {}
        # if self.path.get('alphafold') is None:
        #     messages.append("Path to phenix executables not set. Check in settings!")
        # else:
        #     results['alphafold_path'] = self.path.get('alphafold')
        #     self.update_path(self.path.get('alphafold'), 'alphafold', sess)
        # return messages

    def add_blank_entry(self, sess):
        settings = self.get_from_db(sess)
        if settings is None:
            logger.debug("Adding first entry to settings")
            blank = {}
            for var in vars(self):
                logger.debug(var)
                obj = getattr(self, var)
                #try:
                #    if hasattr(obj, db):
                #        if obj.db and not obj.db_primary_key:
                #            blank[obj] = ''
                #except NameError:
                #    pass
                #except Exception as e:
                #    traceback.print_exc()
            #blank.update({'global_config': False})
            self.db_insert_settings([blank], sess)
            return True
        else:
            return False


    def db_insert_settings(self, data, sess):
        logger.debug("Insert settings into DB")
        logger.debug(data)
        assert isinstance(data, list)
        rows = [self.db.Settings(**row) for row in data]
        for row in rows:
            sess.merge(row)
        sess.commit()

    # def update_queue_submit(self, value, sess):
    #     result = sess.query(self.db.Settings).get(1)
    #     result.queue_submit_dialog = value
    #     sess.commit()

    def update_settings(self, insert_dict, sess):
        logger.debug("Update settings")
        settings = self.get_from_db(sess)
        if settings is None:
            self.db_insert_settings(insert_dict, sess)
        else:
            for key, value in insert_dict[0].items():
                setattr(settings, key, value)
                logger.debug(f"Setting {key} to {value} type {type(value)}")
            sess.commit()

class DefaultValues:
    def __init__(self, other):
        self.job_name = None
        self.output_dir = None
        self.pipeline = 'full'
        self.entity_list = None
        self.protein_entitites = None
        self.dna_entities = None
        self.rna_entities = None
        self.bond_entities = None
        self.ligand_entities = None
        _, project_id = other.prj.get_active_project(other.sess)
        if project_id:
            self.precomputed_msas_path = other.prj.get_path_by_project_id(project_id, other.sess)
        settings = other.settings.get_from_db(other.sess)
        #update with defaults from advanced_settings
        advanced_settings_defaults = AdvancedSettingsDefaults()
        self.__dict__.update(advanced_settings_defaults.__dict__)
        logging.debug("Default values:")
        logging.debug(self.__dict__)
        if settings.queue_default:
            self.queue = True
