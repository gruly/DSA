#!/usr/bin/perl

use utf8;
use Encode qw(decode_utf8);
binmode STDOUT, ":utf8";

use Getopt::Long;
use File::Basename;

my $_VERBOSE=0;
my $_HELP;


my %_CONFIG =(
'verbose' => \$_VERBOSE,
'help' => \$_HELP
);

my $usage = "  \
--verbose|--v       : verbose \
--help|h            :   \n";

######################## Parsing parameters with GetOptions
$_HELP = 1 unless GetOptions(\%_CONFIG,'verbose|v', 'help|h');
die "Not enough arguments\n$usage" unless(scalar @ARGV == 2);
#######################################################################"
my $in = $ARGV[0];
my $out = $ARGV[1];
open(F1,$File1) or die "Can't open file1 \n"; #  ctm
my @FILE = <F1>;
close(F1);

unless (open (OUT, "> $out"))
   {
                 print OUT ("Erreur de  $out");
  }




%map =('[' => 'pers', '(' => 'func', '{' => 'org', '$' => 'loc', '&' => 'prod', '%' => 'amount', '#' => 'time', ')' => 'event' );
# ']' => 'end'
for ($j=0; $j< @FILE; $j++){
	$ligne=$FILE[$j];
        chomp($ligne);
        @words2=split(" ",$ligne);
        $dne="";
        $value="";
        $end="";
        for($i=0; $i<scalar(@words2); $i++){
                if (  %map{$words2[$i]} ne "" ){
                        $dne=$words2[$i];
                        $value.=$dne." ";
                        $end="";
                }
                elsif ($words2[$i] eq ']' and $dne ne "" ){
                        $value.=$words2[$i]." ";
                        $dne="";
                        $end=']';
                }
                elsif (%map{$words2[$i]} eq ""  and  $dne ne "" and $end eq "" ){
                        $value.=$words2[$i]." ";

                }
                elsif ($words2[$i] eq ']' and $dne eq ""){#traiter les cas particulier
                        $value.=$words2[$i]." ";
                        $end="]";
                        $dne="";
                }
                else{
                        $value.="* ";
                }
        }
        if ($dne ne ""){
                $value.=$dne;
        }
        print $value," ",$sent_id ,"\n";


}



