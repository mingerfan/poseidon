"""Local Agent source transaction; no compiler, runtime or Git mutation.

The caller must exclude project execution and hold the existing SDK/native
locks. Journals and original bytes remain separate from historical plans.
"""
import hashlib,json,os,stat,time,uuid
from pathlib import Path

def require(value,message):
    if not value:raise ValueError(message)

def sha(path):
    p=Path(path)
    require(p.is_file() and not p.is_symlink(),"Regular source")
    return hashlib.sha256(p.read_bytes()).hexdigest()

def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def atomic_json(path,value):
    path=Path(path);temporary=path.with_name("."+path.name+"."+uuid.uuid4().hex+".tmp")
    with temporary.open("xb") as stream:
        stream.write(json.dumps(value,sort_keys=True,separators=(",",":"),allow_nan=False).encode())
        os.fchmod(stream.fileno(),0o600);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path);fsync_directory(path.parent)

def regular_under(root,relative):
    rel=Path(relative)
    require(not rel.is_absolute() and rel.parts and ".." not in rel.parts,"Relative source path")
    path=root/rel
    require(path.parent==path.parent.resolve() and path.parent.is_relative_to(root) and not path.is_symlink(),"Source remains inside checkout")
    return path

def observed_original(path,item):
    original=item["original_sha256"]
    if original is None:
        require(not os.path.lexists(path),"New source collision")
    else:
        require(sha(path)==original and stat.S_IMODE(path.stat().st_mode)==item["original_mode"],"Preserve current source and permission")

def apply_transaction(root,targets,recovery,observe,context,finalize=lambda:None):
    """Replace reviewed files; restore only bytes still owned by this transaction."""
    root=Path(root);recovery=Path(recovery)
    require(root==root.resolve() and root.is_dir(),"Resolved checkout")
    require(recovery.parent==recovery.parent.resolve() and recovery.parent.is_relative_to(root) and not os.path.lexists(recovery),"Fresh checkout recovery directory")
    require(callable(observe) and callable(finalize) and type(targets) is dict and targets,"Reviewed transaction")
    observe()
    rows=[]
    for relative,item in targets.items():
        require(set(item)=={"source","target_sha256","original_sha256","original_mode","target_mode"},"Reviewed source fields")
        target=regular_under(root,relative)
        source=Path(item["source"])
        require(source.resolve().is_relative_to(root) and source.is_file() and not source.is_symlink(),"Reviewed regular target")
        require(sha(source)==item["target_sha256"],"Reviewed target bytes")
        require(type(item["target_mode"]) is int and 0<=item["target_mode"]<=0o777,"Target permission")
        observed_original(target,item)
        if item["original_sha256"] is not None:
            require(item["target_mode"]==item["original_mode"],"Preserve original permission")
        else:require(item["target_mode"]==0o644,"New module permission")
        rows.append((relative,target,source,item))
    require(len({target.name for _,target,_,_ in rows})==len(rows),"Unique target basenames")
    require(all(target.parent.stat().st_dev==recovery.parent.stat().st_dev for _,target,_,_ in rows),
            "Source and recovery must share one filesystem")
    recovery.mkdir(mode=0o700);fsync_directory(recovery.parent)
    (recovery/"originals").mkdir(mode=0o700)
    (recovery/"install-displaced").mkdir(mode=0o700)
    journal=dict(format="poseidon-agent-source-transaction-v1",state="preparing",
      context=context,root=str(root),targets=targets,installed=[],restored=[],
      rollback_conflicts=[],retained_files=[],rollback_intents={},
       install_intents={},install_conflicts=[],created_wall=time.time())
    journal_path=recovery/"journal.json"
    atomic_json(journal_path,journal)
    prepared=[]
    try:
        for relative,target,source,item in rows:
            if item["original_sha256"] is not None:
                backup=recovery/"originals"/target.name
                with backup.open("xb") as stream:
                    stream.write(target.read_bytes());os.fchmod(stream.fileno(),item["original_mode"])
                    stream.flush();os.fsync(stream.fileno())
                require(sha(backup)==item["original_sha256"],"Exact recovery bytes")
            temporary=target.with_name("."+target.name+"."+uuid.uuid4().hex+".tmp")
            with temporary.open("xb") as stream:
                prepared.append((relative,target,temporary,item))
                stream.write(source.read_bytes());os.fchmod(stream.fileno(),item["target_mode"])
                stream.flush();os.fsync(stream.fileno())
            require(sha(temporary)==item["target_sha256"],"Prepared target bytes")
        fsync_directory(recovery/"originals")
        for directory in {target.parent for _,target,_,_ in prepared}:fsync_directory(directory)
        journal["state"]="prepared";atomic_json(journal_path,journal)
        observe()
        for relative,target,temporary,item in prepared:
            observe()
            retained=None
            intent=dict(state="preparing",retained=None,prepared=str(temporary))
            journal["install_intents"][relative]=intent
            journal.update(state="installing",next_target=relative)
            atomic_json(journal_path,journal)
            try:
                target=regular_under(root,relative)
                if item["original_sha256"] is not None:
                    displaced_dir=recovery/"install-displaced"/target.name
                    displaced_dir.mkdir(mode=0o700);fsync_directory(displaced_dir.parent)
                    retained=displaced_dir/"current"
                    intent.update(state="displacing",retained=str(retained))
                    atomic_json(journal_path,journal)
                    os.rename(target,retained)
                    journal["retained_files"].append(str(retained))
                    fsync_directory(target.parent);fsync_directory(displaced_dir)
                    intent["state"]="displaced";atomic_json(journal_path,journal)
                    require(sha(retained)==item["original_sha256"] and
                      stat.S_IMODE(retained.stat().st_mode)==item["original_mode"],
                      "Captured current source changed; preserve external bytes")
                intent["state"]="publishing";atomic_json(journal_path,journal)
                # Exclusive publication also protects originally absent paths.
                os.link(temporary,target,follow_symlinks=False)
                journal["installed"].append(relative)
                temporary.unlink();fsync_directory(target.parent)
                require(sha(target)==item["target_sha256"] and
                  stat.S_IMODE(target.stat().st_mode)==item["target_mode"],"Installed source identity")
                intent["state"]="installed";atomic_json(journal_path,journal)
            except BaseException as install_error:
                republished=False;publish_error=None;external=False
                if relative not in journal["installed"]:
                    if retained is not None and os.path.lexists(retained):
                        if str(retained) not in journal["retained_files"]:
                            journal["retained_files"].append(str(retained))
                        try:
                            external=not (sha(retained)==item["original_sha256"] and
                              stat.S_IMODE(retained.stat().st_mode)==item["original_mode"])
                        except BaseException:external=True
                        try:
                            os.link(retained,target,follow_symlinks=False)
                            fsync_directory(target.parent);republished=True
                        except BaseException as publish:
                            publish_error=type(publish).__name__
                            if os.path.lexists(target):external=True
                    elif item["original_sha256"] is not None:
                        try:observed_original(target,item)
                        except BaseException:external=True
                    elif os.path.lexists(target):external=True
                if external:
                    journal["install_conflicts"].append(dict(target=relative,error_type=type(install_error).__name__))
                intent.update(state="failed",error_type=type(install_error).__name__,
                              current_republished=republished,republish_error_type=publish_error)
                try:atomic_json(journal_path,journal)
                except BaseException:pass
                raise
        observe();finalize()
        journal.pop("next_target",None)
        journal.update(state="applied",completed_wall=time.time())
        atomic_json(journal_path,journal)
        return journal
    except BaseException as error:
        journal.update(state="rollback",failure_type=type(error).__name__)
        # Recovery errors must not replace the exception that caused rollback.
        try:
            atomic_json(journal_path,journal)
            displaced_root=recovery/"displaced"
            restore_root=recovery/"restores"
            displaced_root.mkdir(mode=0o700);restore_root.mkdir(mode=0o700)
            fsync_directory(recovery)
            for relative in reversed(journal["installed"]):
                item=targets[relative];retained=None;intent=None
                try:
                    # Path conflicts belong in the journal, including symlinks.
                    target=regular_under(root,relative)
                    restore=None
                    if item["original_sha256"] is not None:
                        backup=recovery/"originals"/target.name
                        require(sha(backup)==item["original_sha256"],"Exact rollback original")
                        restore=restore_root/(target.name+".restore")
                        with restore.open("xb") as stream:
                            stream.write(backup.read_bytes());stream.flush();os.fsync(stream.fileno())
                        os.chmod(restore,item["original_mode"])
                        with restore.open("rb") as stream:os.fsync(stream.fileno())
                        require(sha(restore)==item["original_sha256"] and
                          stat.S_IMODE(restore.stat().st_mode)==item["original_mode"],"Exact prepared rollback original")
                        fsync_directory(restore_root)
                    # Each move has a new private destination; never replace
                    # prior recovery evidence. The intent precedes the rename.
                    displaced_dir=displaced_root/target.name
                    displaced_dir.mkdir(mode=0o700);fsync_directory(displaced_dir.parent)
                    retained=displaced_dir/"current"
                    intent=dict(state="displacing",retained=str(retained),
                                restore=None if restore is None else str(restore))
                    journal["rollback_intents"][relative]=intent
                    atomic_json(journal_path,journal)
                    os.rename(target,retained)
                    journal["retained_files"].append(str(retained))
                    fsync_directory(target.parent);fsync_directory(displaced_dir)
                    intent["state"]="displaced";atomic_json(journal_path,journal)
                    # Inspect the bytes actually displaced, rather than bytes
                    # observed before a later destructive replacement.
                    require(sha(retained)==item["target_sha256"] and
                      stat.S_IMODE(retained.stat().st_mode)==item["target_mode"],
                      "External source modification; retain user bytes")
                    if restore is not None:
                        intent["state"]="publishing_original";atomic_json(journal_path,journal)
                        # link() creates a new path and fails if an external
                        # writer rebuilt it. It never overwrites that writer.
                        os.link(restore,target,follow_symlinks=False)
                        fsync_directory(target.parent)
                        require(sha(target)==item["original_sha256"] and
                          stat.S_IMODE(target.stat().st_mode)==item["original_mode"],"Published rollback original changed")
                    else:
                        require(not os.path.lexists(target),"New source path recreated during rollback")
                    journal["restored"].append(relative);intent["state"]="restored"
                except BaseException as conflict:
                    # Put displaced current bytes back only when the path is
                    # still absent. Even a dangling symlink is a collision.
                    republished=False;publish_error=None
                    if retained is not None and os.path.lexists(retained):
                        if str(retained) not in journal["retained_files"]:
                            journal["retained_files"].append(str(retained))
                        try:
                            os.link(retained,target,follow_symlinks=False)
                            fsync_directory(target.parent);republished=True
                        except BaseException as publish:
                            publish_error=type(publish).__name__
                    journal["rollback_conflicts"].append(dict(target=relative,
                      error_type=type(conflict).__name__,current_republished=republished,
                      republish_error_type=publish_error))
                    if intent is not None:
                        intent.update(state="conflict",error_type=type(conflict).__name__,
                                      current_republished=republished,republish_error_type=publish_error)
                atomic_json(journal_path,journal)
            for relative,target,temporary,item in prepared:
                if os.path.lexists(temporary):
                    retained=recovery/("prepared-"+target.name)
                    require(not os.path.lexists(retained),"Prepared recovery evidence collision")
                    os.rename(temporary,retained);journal["retained_files"].append(str(retained))
                    fsync_directory(temporary.parent)
            journal.pop("next_target",None)
            journal.update(state="rollback_conflict" if journal["rollback_conflicts"] or journal["install_conflicts"] else "rolled_back",completed_wall=time.time())
            atomic_json(journal_path,journal)
        except BaseException as recovery_error:
            journal.update(state="rollback_conflict",recovery_failure_type=type(recovery_error).__name__)
            try:atomic_json(journal_path,journal)
            except BaseException:pass
        raise
