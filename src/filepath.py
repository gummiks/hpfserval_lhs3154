import os 

class FilePath(object):
    """
    A simple filepath class to deal with files

    NOTES:
    """
    def __init__(self,fullpath):
        self._fullpath = os.path.abspath(fullpath)

    @property
    def basename(self):
        return self._fullpath.split(os.sep)[-1]

    @property
    def barename(self):
        return ".".join(self.basename.split(".")[0:-1])

    @property 
    def extension(self):
        return self._fullpath.split(".")[-1]

    @extension.setter
    def extension(self,value):
        self._fullpath = self.directory + os.sep + self.barename + "." + value

    @property
    def directory(self):
        return os.sep.join(self._fullpath.split(os.sep)[:-1])

    @directory.setter
    def directory(self,value):
        if value[-1]==os.sep:
            self._fullpath = value + self.basename
        else:
            self._fullpath = value + os.sep + self.basename

    def __str__(self):
        return self._fullpath

    def add_prefix(self,value):
        self._fullpath = self.directory + os.sep + value + self.basename 

    def add_suffix(self,value):
        self._fullpath = self.directory + os.sep + self.barename + value + "." + self.extension

    def __repr__(self):
        return '"' + self._fullpath + '"'

