import uuid
import random
from abc import ABC, abstractmethod

import config
from controllers.controller import Controller


class Module(ABC):
    MAX_BODY_CHILDREN = 1
    MAX_LIMB_CHILDREN = 2

    def __init__(self, name: str, parent, con_site: int, angle: int,
                 controller_class: type[Controller], init: bool = False):
        self.name = name
        self.parent = parent
        self.connection_site = con_site
        self.angle = angle
        self.controller_class = controller_class
        self.controller = controller_class(name, parent.controller if parent is not None else None, init)
        self.children = []
        self.number_of_limb_children = 0
        self.number_of_body_children = 0

    @abstractmethod
    def get_dict_for_json(self) -> dict:
        pass

    @abstractmethod
    def add_limb(self, init: bool = False):
        pass

    def can_add_limb(self) -> bool:
        return self.number_of_limb_children < self.MAX_LIMB_CHILDREN

    def can_add_body(self) -> bool:
        return self.number_of_body_children < self.MAX_BODY_CHILDREN
    
    def DFS(self, modules: list):
        modules.append(self)
        for child in self.children:
            if child.connection_site == 2:
                child.DFS(modules)
                return


class BodyJoint(Module):
    MAX_BODY_CHILDREN = 1
    MAX_LIMB_CHILDREN = 2

    def __init__(self, name: str, parent: Module, con_site: int, angle: int,
                 controller_class: type[Controller], joint_type: str, init: bool = False):
        super().__init__(name, parent, con_site, angle, controller_class, init)
        self.joint_type = joint_type

    def get_dict_for_json(self) -> dict:
        return {
            "name": self.name,
            "parent": self.parent.name,
            "connection_site": str(self.connection_site),
            "type": self.joint_type,
            "angle": self.angle,
            "rgb": [0.0, 0.0, 0.0]
        }

    def add_limb(self, init: bool = False):  # Adds a pair of limb children
        if self.can_add_limb():
            self.number_of_limb_children += 2
            limb_type = random.choice(config.LIMB_JOINTS)
            angle = random.choice(config.ROTATIONS)
            limb1 = LimbJoint(str(uuid.uuid4()), self, 0, angle, self.controller_class,
                              limb_type, init)
            limb2 = LimbJoint(str(uuid.uuid4()), self, 1, -angle, self.controller_class,
                              limb_type, init)
            limb1.complementary_limb = limb2
            limb2.complementary_limb = limb1
            self.children.append(limb1)
            self.children.append(limb2)

    def add_body(self, init: bool = False):
        if self.can_add_body():
            self.number_of_body_children += 1
            body_type = random.choice(config.BODY_JOINTS)
            body = BodyJoint(str(uuid.uuid4()), self, 2, 0, self.controller_class, body_type, init)
            self.children.append(body)

    def swap(self):
        other_joint_types = config.BODY_JOINTS[:]
        other_joint_types.remove(self.joint_type)
        self.joint_type = random.choice(other_joint_types)


class Root(BodyJoint):
    MAX_BODY_CHILDREN = 2
    MAX_LIMB_CHILDREN = 2

    def __init__(self, controller_class: type[Controller]):
        super().__init__("root", None, 0, 0, controller_class, "Root", init=True)

    def get_dict_for_json(self) -> dict:
        return {
            "name": self.name,
            "parent": "",
            "connection_site": self.connection_site,
            "type": self.joint_type,
            "angle": self.angle,
            "rgb": [0.0, 0.0, 0.0]
        }
    
    def add_body(self, init : bool = False):
        if self.can_add_body():
            con_site = 2
            for child in self.children:
                if child.connection_site == con_site:
                    con_site = 3  # 2 or 3 is free because we can add body
            self.number_of_body_children += 1
            body_type = random.choice(config.BODY_JOINTS)
            body = BodyJoint(str(uuid.uuid4()), self, con_site, 0,
                             self.controller_class, body_type, init)
            self.children.append(body)


class LimbJoint(Module):
    MAX_BODY_CHILDREN = 0
    MAX_LIMB_CHILDREN = 1

    def __init__(self, name: str, parent: Module, con_site: int, angle: int,
                 controller_class: type[Controller], joint_type: str, init: bool = False):
        super().__init__(name, parent, con_site, angle, controller_class, init)
        self.joint_type = joint_type
        self.complementary_limb = None

    def get_dict_for_json(self) -> dict:
        return {
            "name": self.name,
            "parent": self.parent.name,
            "connection_site": str(self.connection_site),
            "type": self.joint_type,
            "angle": self.angle,
            "rgb": [0.0, 0.0, 0.0]
        }

    # Adds a pair of limb children
    def add_limb(self, init: bool = False):
        if self.can_add_limb() and self.complementary_limb is not None:
            self.number_of_limb_children += 1
            self.complementary_limb.number_of_limb_children += 1
            con_site = random.choice((0, 1, 2))
            angle = random.choice(config.ROTATIONS)
            if con_site == 2:
                compl_angle = -angle
            else:
                compl_angle = -angle + 180
            limb_type = random.choice(config.LIMB_JOINTS)
            limb1 = LimbJoint(str(uuid.uuid4()), self, con_site, angle,
                              self.controller_class, limb_type, init)
            limb2 = LimbJoint(str(uuid.uuid4()), self.complementary_limb, con_site, compl_angle,
                              self.controller_class, limb_type, init)
            limb1.complementary_limb = limb2
            limb2.complementary_limb = limb1
            self.children.append(limb1)
            self.complementary_limb.children.append(limb2)

    def swap(self):
        other_joint_types = config.LIMB_JOINTS[:]
        other_joint_types.remove(self.joint_type)
        self.joint_type = random.choice(other_joint_types)
        self.complementary_limb.joint_type = self.joint_type
        self.angle = random.choice(config.ROTATIONS)
        if type(self.parent) == LimbJoint:
            self.connection_site = random.choice((0, 1, 2))
            self.complementary_limb.connection_site = self.connection_site
            if self.connection_site == 2:
                self.complementary_limb.angle = -self.angle
            else:
                self.complementary_limb.angle = -self.angle + 180
        else:
            self.complementary_limb.angle = -self.angle

    def DFS_count(self, n: int):
        n += 1
        if len(self.children) != 0:
            return self.children[0].DFS_count(n)
        return n
