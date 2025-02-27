##warning: very bad attempt. Do not try!

#set the atoms and num_of_frames in MD Simulations
set num_frames [molinfo top get numframes]

#prepare matrix to store the rmsd
set rmsd_matrix {}

#nested loops (from bottom to top)
for {set ref_frame 0} {$ref_frame < $num_frames} {incr ref_frame} {
##   animate goto $ref_frame
puts $ref_frame
   set rmsd_list {}
   set ref_str [atomselect top "protein" frame $ref_frame]
     for {set fit_frame 0} {$fit_frame < $num_frames} {incr fit_frame} {
      set fit_str [atomselect top "protein" frame $fit_frame]
      set trans_mat [measure fit $fit_str $ref_str]
      $fit_str move $trans_mat
      set rmsd [measure rmsd $ref_str $fit_str]
      lappend rmsd_list $rmsd
       }
   lappend rmsd_matrix $rmsd_list
}

set file_id [open "rmsd_matrix.txt" "w"]
foreach row $rmsd_matrix {
  puts $file_id [join $row "\t"]
}

close $file_id
