"""Transactions use temporary checkouts; production and backend remain untouched."""
import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import source_transaction as t

class TransactionTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
  self.root=Path(self.temp.name).resolve();self.source=self.root/"reviewed";self.source.mkdir()
  self.dest=self.root/"production";self.dest.mkdir();(self.root/"recovery").mkdir()
  self.recovery=self.root/"recovery"/"one"
  self.targets={}
  for name,old,mode in [("a.py",b"old-a",0o664),("new.py",None,0o644),("b.py",b"old-b",0o644)]:
   p=self.dest/name;s=self.source/name;s.write_bytes(b"new-"+name.encode())
   if old is not None:p.write_bytes(old);p.chmod(mode)
   self.targets[str(p.relative_to(self.root))]=dict(source=str(s),target_sha256=t.sha(s),
     original_sha256=None if old is None else t.sha(p),original_mode=None if old is None else mode,target_mode=mode)
 def apply(self,observe=lambda:None):
  return t.apply_transaction(self.root,self.targets,self.recovery,observe,{"qualification_binding":"test-only"})
 def journal(self):return json.loads((self.recovery/"journal.json").read_text())
 def test_success_preserves_original_bytes_and_permissions(self):
  r=self.apply()
  self.assertEqual(r["state"],"applied");self.assertEqual(len(r["installed"]),3)
  self.assertEqual((self.recovery/"originals/a.py").read_bytes(),b"old-a")
  self.assertEqual((self.dest/"a.py").stat().st_mode&0o777,0o664)
  self.assertEqual((self.dest/"new.py").stat().st_mode&0o777,0o644)
 def test_changed_user_source_rejected_before_recovery_or_replacement(self):
  (self.dest/"a.py").write_bytes(b"user")
  with self.assertRaisesRegex(ValueError,"Preserve current"):self.apply()
  self.assertEqual((self.dest/"a.py").read_bytes(),b"user");self.assertFalse(self.recovery.exists())
 def test_dangling_new_file_symlink_rejected_without_overwrite(self):
  (self.dest/"new.py").symlink_to(self.dest/"missing")
  with self.assertRaisesRegex(ValueError,"inside checkout"):self.apply()
  self.assertTrue((self.dest/"new.py").is_symlink());self.assertFalse(self.recovery.exists())
 def test_existing_new_file_collision_preserved(self):
  (self.dest/"new.py").write_bytes(b"user")
  with self.assertRaisesRegex(ValueError,"collision"):self.apply()
  self.assertEqual((self.dest/"new.py").read_bytes(),b"user")
 def test_original_permission_is_not_normalized_to_644(self):
  self.targets["production/a.py"]["target_mode"]=0o644
  with self.assertRaisesRegex(ValueError,"original permission"):self.apply()
  self.assertEqual((self.dest/"a.py").stat().st_mode&0o777,0o664)
 def test_target_bytes_changed_after_review_are_rejected(self):
  (self.source/"a.py").write_bytes(b"changed")
  with self.assertRaisesRegex(ValueError,"Reviewed target bytes"):self.apply()
  self.assertEqual((self.dest/"a.py").read_bytes(),b"old-a")
 def test_quiescence_failure_prevents_any_file_change(self):
  def busy():raise ValueError("active request")
  with self.assertRaisesRegex(ValueError,"active request"):self.apply(busy)
  self.assertFalse(self.recovery.exists());self.assertEqual((self.dest/"a.py").read_bytes(),b"old-a")
 def test_partial_failure_restores_old_files_and_retains_new_file_evidence(self):
  link=t.os.link
  def fail(source,target,**kwargs):
   if Path(target)==self.dest/"b.py" and str(source).endswith(".tmp"):raise OSError("injected")
   return link(source,target,**kwargs)
  with patch.object(t.os,"link",side_effect=fail):
   with self.assertRaisesRegex(OSError,"injected"):self.apply()
  self.assertEqual(self.journal()["state"],"rolled_back")
  self.assertEqual((self.dest/"a.py").read_bytes(),b"old-a")
  self.assertEqual((self.dest/"a.py").stat().st_mode&0o777,0o664)
  self.assertEqual((self.dest/"b.py").read_bytes(),b"old-b")
  self.assertFalse((self.dest/"new.py").exists())
  self.assertEqual((self.recovery/"displaced/new.py/current").read_bytes(),b"new-new.py")
 def test_failure_after_displacement_before_directory_fsync_preserves_original(self):
  real=t.fsync_directory;failed=[]
  def fail_once(path):
   if Path(path)==self.dest and not failed:failed.append(True);raise OSError("fsync interrupted")
   return real(path)
  with patch.object(t,"fsync_directory",side_effect=fail_once):
   with self.assertRaisesRegex(OSError,"fsync interrupted"):self.apply()
  self.assertEqual((self.dest/"a.py").read_bytes(),b"old-a")
  self.assertEqual(self.journal()["state"],"rolled_back")
 def test_concurrent_external_edit_is_preserved_during_rollback(self):
  def observe():
   if (self.dest/"a.py").read_bytes()==b"new-a.py":
    (self.dest/"a.py").write_bytes(b"external edit");raise ValueError("new active request")
  with self.assertRaisesRegex(ValueError,"active request"):self.apply(observe)
  self.assertEqual((self.dest/"a.py").read_bytes(),b"external edit")
  self.assertEqual(self.journal()["state"],"rollback_conflict")
  self.assertEqual(self.journal()["rollback_conflicts"][0]["target"],"production/a.py")
  self.assertEqual((self.recovery/"originals/a.py").read_bytes(),b"old-a")
 def test_outside_checkout_target_rejected(self):
  item=self.targets.pop("production/a.py");self.targets["../outside.py"]=item
  with self.assertRaisesRegex(ValueError,"Relative source path"):self.apply()
  self.assertFalse(self.recovery.exists())
 def test_failed_final_inventory_check_restores_every_target(self):
  def fail():raise ValueError("final inventory")
  with self.assertRaisesRegex(ValueError,"final inventory"):
   t.apply_transaction(self.root,self.targets,self.recovery,lambda:None,{"test_only":True},fail)
  self.assertEqual(self.journal()["state"],"rolled_back")
  self.assertEqual((self.dest/"a.py").read_bytes(),b"old-a")
  self.assertEqual((self.dest/"b.py").read_bytes(),b"old-b")
  self.assertFalse((self.dest/"new.py").exists())
 def test_existing_recovery_record_is_not_overwritten(self):
  self.recovery.mkdir();(self.recovery/"note").write_text("retain")
  with self.assertRaisesRegex(ValueError,"Fresh checkout recovery"):self.apply()
  self.assertEqual((self.recovery/"note").read_text(),"retain")

 def failed_finalize(self):
  raise RuntimeError("original finalize failure")
 def finalize_failure(self):
  return t.apply_transaction(self.root,self.targets,self.recovery,lambda:None,
                            {"test_only":True},self.failed_finalize)
 def test_external_edit_while_restore_is_prepared_is_not_overwritten(self):
  chmod=t.os.chmod;edited=[]
  def edit(path,mode):
   if Path(path).name=="a.py.restore":
    (self.dest/"a.py").write_bytes(b"edit while preparing restore");edited.append(True)
   return chmod(path,mode)
  with patch.object(t.os,"chmod",side_effect=edit):
   with self.assertRaisesRegex(RuntimeError,"original finalize failure"):self.finalize_failure()
  self.assertTrue(edited)
  self.assertEqual((self.dest/"a.py").read_bytes(),b"edit while preparing restore")
  self.assertEqual((self.recovery/"displaced/a.py/current").read_bytes(),b"edit while preparing restore")
  self.assertEqual(self.journal()["state"],"rollback_conflict")
  self.assertEqual(self.journal()["rollback_intents"]["production/a.py"]["state"],"conflict")
 def test_external_edit_at_atomic_displacement_is_not_overwritten(self):
  rename=t.os.rename;edited=[]
  def edit(source,destination):
   if Path(source)==self.dest/"a.py" and Path(destination).is_relative_to(self.recovery/"displaced"):
    Path(source).write_bytes(b"edit immediately before displacement");edited.append(True)
   return rename(source,destination)
  with patch.object(t.os,"rename",side_effect=edit):
   with self.assertRaisesRegex(RuntimeError,"original finalize failure"):self.finalize_failure()
  self.assertTrue(edited)
  self.assertEqual((self.dest/"a.py").read_bytes(),b"edit immediately before displacement")
  self.assertEqual((self.recovery/"displaced/a.py/current").read_bytes(),b"edit immediately before displacement")
  self.assertEqual(self.journal()["state"],"rollback_conflict")
 def test_recreated_path_at_restore_publish_is_not_overwritten(self):
  link=t.os.link;created=[]
  def create(source,destination,**kwargs):
   if Path(source)==self.recovery/"restores/a.py.restore":
    Path(destination).write_bytes(b"external recreated path");created.append(True)
   return link(source,destination,**kwargs)
  with patch.object(t.os,"link",side_effect=create):
   with self.assertRaisesRegex(RuntimeError,"original finalize failure"):self.finalize_failure()
  self.assertTrue(created)
  self.assertEqual((self.dest/"a.py").read_bytes(),b"external recreated path")
  self.assertEqual((self.recovery/"displaced/a.py/current").read_bytes(),b"new-a.py")
  self.assertEqual(self.journal()["state"],"rollback_conflict")
  conflict=next(x for x in self.journal()["rollback_conflicts"] if x["target"]=="production/a.py")
  self.assertEqual(conflict["error_type"],"FileExistsError")
 def test_dangling_symlink_recreated_at_restore_publish_is_preserved(self):
  link=t.os.link;missing=self.dest/"missing-external"
  def create(source,destination,**kwargs):
   if Path(source)==self.recovery/"restores/a.py.restore":
    Path(destination).symlink_to(missing)
   return link(source,destination,**kwargs)
  with patch.object(t.os,"link",side_effect=create):
   with self.assertRaisesRegex(RuntimeError,"original finalize failure"):self.finalize_failure()
  self.assertTrue((self.dest/"a.py").is_symlink())
  self.assertEqual(os.readlink(self.dest/"a.py"),str(missing))
  self.assertEqual(self.journal()["state"],"rollback_conflict")
 def test_existing_symlink_conflict_records_failure_and_preserves_original_exception(self):
  missing=self.dest/"missing-external"
  def fail():
   target=self.dest/"a.py";target.unlink();target.symlink_to(missing)
   raise RuntimeError("original symlink interruption")
  with self.assertRaisesRegex(RuntimeError,"original symlink interruption"):
   t.apply_transaction(self.root,self.targets,self.recovery,lambda:None,{"test_only":True},fail)
  self.assertTrue((self.dest/"a.py").is_symlink())
  journal=self.journal()
  self.assertEqual(journal["state"],"rollback_conflict")
  self.assertEqual(journal["failure_type"],"RuntimeError")
  self.assertIn("production/a.py",[x["target"] for x in journal["rollback_conflicts"]])
  self.assertEqual((self.dest/"b.py").read_bytes(),b"old-b")
  self.assertFalse((self.dest/"new.py").exists())
 def test_symlink_inserted_at_displacement_is_retained_and_republished(self):
  rename=t.os.rename;missing=self.dest/"missing-external"
  def substitute(source,destination):
   if Path(source)==self.dest/"a.py" and Path(destination).is_relative_to(self.recovery/"displaced"):
    Path(source).unlink();Path(source).symlink_to(missing)
   return rename(source,destination)
  with patch.object(t.os,"rename",side_effect=substitute):
   with self.assertRaisesRegex(RuntimeError,"original finalize failure"):self.finalize_failure()
  target=self.dest/"a.py";retained=self.recovery/"displaced/a.py/current"
  self.assertTrue(target.is_symlink());self.assertTrue(retained.is_symlink())
  self.assertEqual(os.readlink(target),str(missing))
  conflict=next(x for x in self.journal()["rollback_conflicts"] if x["target"]=="production/a.py")
  self.assertTrue(conflict["current_republished"])
 def test_displacement_and_publish_intents_precede_filesystem_changes(self):
  rename=t.os.rename;link=t.os.link;observed=[]
  def inspect_move(source,destination):
   if Path(source)==self.dest/"a.py" and Path(destination).is_relative_to(self.recovery/"displaced"):
    value=self.journal();intent=value["rollback_intents"]["production/a.py"]
    self.assertEqual(value["state"],"rollback")
    self.assertEqual(intent["state"],"displacing")
    self.assertEqual(intent["retained"],str(destination));observed.append("displace")
   return rename(source,destination)
  def inspect_publish(source,destination,**kwargs):
   if Path(source)==self.recovery/"restores/a.py.restore":
    intent=self.journal()["rollback_intents"]["production/a.py"]
    self.assertEqual(intent["state"],"publishing_original");observed.append("publish")
   return link(source,destination,**kwargs)
  with patch.object(t.os,"rename",side_effect=inspect_move),patch.object(t.os,"link",side_effect=inspect_publish):
   with self.assertRaisesRegex(RuntimeError,"original finalize failure"):self.finalize_failure()
  self.assertEqual(observed,["displace","publish"])
  self.assertEqual(self.journal()["rollback_intents"]["production/a.py"]["state"],"restored")
 def test_recovery_storage_failure_does_not_mask_original_exception(self):
  atomic=t.atomic_json
  def fail(path,value):
   if value["state"]=="rollback":raise OSError("recovery storage interruption")
   return atomic(path,value)
  with patch.object(t,"atomic_json",side_effect=fail):
   with self.assertRaisesRegex(RuntimeError,"original finalize failure"):self.finalize_failure()
  journal=self.journal()
  self.assertEqual(journal["state"],"rollback_conflict")
  self.assertEqual(journal["failure_type"],"RuntimeError")
  self.assertEqual(journal["recovery_failure_type"],"OSError")

 def test_external_edit_at_install_displacement_is_not_overwritten(self):
  rename=t.os.rename;edited=[]
  def edit(source,destination):
   if Path(source)==self.dest/"a.py" and Path(destination).is_relative_to(self.recovery/"install-displaced"):
    Path(source).write_bytes(b"external edit during install");edited.append(True)
   return rename(source,destination)
  with patch.object(t.os,"rename",side_effect=edit):
   with self.assertRaisesRegex(ValueError,"Captured current source changed"):self.apply()
  self.assertTrue(edited)
  self.assertEqual((self.dest/"a.py").read_bytes(),b"external edit during install")
  self.assertEqual((self.recovery/"install-displaced/a.py/current").read_bytes(),b"external edit during install")
  self.assertEqual((self.recovery/"originals/a.py").read_bytes(),b"old-a")
  journal=self.journal()
  self.assertEqual(journal["state"],"rollback_conflict")
  self.assertEqual(journal["installed"],[])
  self.assertEqual(journal["install_conflicts"][0]["target"],"production/a.py")
 def test_recreated_old_path_at_install_publish_is_not_overwritten(self):
  link=t.os.link
  def create(source,destination,**kwargs):
   if Path(destination)==self.dest/"a.py" and str(source).endswith(".tmp"):
    Path(destination).write_bytes(b"external install path")
   return link(source,destination,**kwargs)
  with patch.object(t.os,"link",side_effect=create):
   with self.assertRaises(FileExistsError):self.apply()
  self.assertEqual((self.dest/"a.py").read_bytes(),b"external install path")
  self.assertEqual((self.recovery/"install-displaced/a.py/current").read_bytes(),b"old-a")
  self.assertEqual(self.journal()["state"],"rollback_conflict")
 def test_recreated_new_path_at_install_publish_is_not_overwritten(self):
  link=t.os.link
  def create(source,destination,**kwargs):
   if Path(destination)==self.dest/"new.py" and str(source).endswith(".tmp"):
    Path(destination).write_bytes(b"external new path")
   return link(source,destination,**kwargs)
  with patch.object(t.os,"link",side_effect=create):
   with self.assertRaises(FileExistsError):self.apply()
  self.assertEqual((self.dest/"new.py").read_bytes(),b"external new path")
  self.assertEqual((self.dest/"a.py").read_bytes(),b"old-a")
  self.assertEqual(self.journal()["state"],"rollback_conflict")
  self.assertEqual(self.journal()["restored"],["production/a.py"])
 def test_symlink_recreated_at_install_publish_is_preserved(self):
  link=t.os.link;missing=self.dest/"missing-external"
  def create(source,destination,**kwargs):
   if Path(destination)==self.dest/"a.py" and str(source).endswith(".tmp"):
    Path(destination).symlink_to(missing)
   return link(source,destination,**kwargs)
  with patch.object(t.os,"link",side_effect=create):
   with self.assertRaises(FileExistsError):self.apply()
  self.assertTrue((self.dest/"a.py").is_symlink())
  self.assertEqual(os.readlink(self.dest/"a.py"),str(missing))
  self.assertEqual((self.recovery/"install-displaced/a.py/current").read_bytes(),b"old-a")
  self.assertEqual(self.journal()["state"],"rollback_conflict")
 def test_success_keeps_captured_originals_and_exact_install_intents(self):
  journal=self.apply()
  self.assertEqual(journal["rollback_intents"],{})
  self.assertEqual(journal["install_conflicts"],[])
  self.assertEqual(set(journal["install_intents"]),set(self.targets))
  self.assertEqual(journal["retained_files"],[
   str(self.recovery/"install-displaced/a.py/current"),
   str(self.recovery/"install-displaced/b.py/current")])
  for relative,intent in journal["install_intents"].items():
   self.assertEqual(set(intent),{"state","retained","prepared"})
   self.assertEqual(intent["state"],"installed")
   if self.targets[relative]["original_sha256"] is None:self.assertIsNone(intent["retained"])
   else:self.assertEqual(t.sha(intent["retained"]),self.targets[relative]["original_sha256"])
 def test_install_intent_is_durable_before_capture_and_publish(self):
  rename=t.os.rename;link=t.os.link;observed=[]
  def inspect_move(source,destination):
   if Path(source)==self.dest/"a.py" and Path(destination).is_relative_to(self.recovery/"install-displaced"):
    intent=self.journal()["install_intents"]["production/a.py"]
    self.assertEqual(intent["state"],"displacing")
    self.assertEqual(intent["retained"],str(destination));observed.append("capture")
   return rename(source,destination)
  def inspect_publish(source,destination,**kwargs):
   if Path(destination)==self.dest/"a.py" and str(source).endswith(".tmp"):
    intent=self.journal()["install_intents"]["production/a.py"]
    self.assertEqual(intent["state"],"publishing")
    self.assertEqual(intent["prepared"],str(source));observed.append("publish")
   return link(source,destination,**kwargs)
  with patch.object(t.os,"rename",side_effect=inspect_move),patch.object(t.os,"link",side_effect=inspect_publish):
   self.apply()
  self.assertEqual(observed,["capture","publish"])

 def test_external_permission_change_at_install_displacement_is_preserved(self):
  rename=t.os.rename
  def edit(source,destination):
   if Path(source)==self.dest/"a.py" and Path(destination).is_relative_to(self.recovery/"install-displaced"):
    Path(source).chmod(0o600)
   return rename(source,destination)
  with patch.object(t.os,"rename",side_effect=edit):
   with self.assertRaisesRegex(ValueError,"Captured current source changed"):self.apply()
  self.assertEqual((self.dest/"a.py").read_bytes(),b"old-a")
  self.assertEqual((self.dest/"a.py").stat().st_mode&0o777,0o600)
  self.assertEqual((self.recovery/"install-displaced/a.py/current").stat().st_mode&0o777,0o600)
  self.assertEqual(self.journal()["state"],"rollback_conflict")
 def test_external_deletion_before_install_capture_is_preserved_and_recorded(self):
  rename=t.os.rename
  def remove(source,destination):
   if Path(source)==self.dest/"a.py" and Path(destination).is_relative_to(self.recovery/"install-displaced"):
    Path(source).unlink()
   return rename(source,destination)
  with patch.object(t.os,"rename",side_effect=remove):
   with self.assertRaises(FileNotFoundError):self.apply()
  self.assertFalse((self.dest/"a.py").exists())
  self.assertEqual(self.journal()["state"],"rollback_conflict")
  self.assertEqual(self.journal()["install_conflicts"][0]["target"],"production/a.py")

if __name__=="__main__":unittest.main(verbosity=2)
