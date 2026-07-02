/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         https://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

// Characterization tests pinning the observable contract of
// itk::GDCMSeriesFileNames (issue #6467). These must remain green when the
// backend is migrated from the deprecated gdcm::SerieHelper to
// gdcm::Scanner + gdcm::IPPSorter: one series identifier for a single-series
// directory, deterministic geometric ordering that reconstructs a valid
// uniformly-spaced volume, GetInputFileNames == first series, and the
// Recursive flag controlling descent.

#include "gtest/gtest.h"
#include "itkGDCMSeriesFileNames.h"
#include "itkGDCMImageIO.h"
#include "itkImageSeriesReader.h"
#include "itkImage.h"
#include "itkMath.h"
#include "itksys/SystemTools.hxx"
#include "itksys/Directory.hxx"
#include "gdcmReader.h"
#include "gdcmWriter.h"
#include "gdcmDataElement.h"
#include "gdcmTag.h"
#include <cstring>
#include <string>
#include <vector>

#define _STRING(s) #s
#define TOSTRING(s) std::string(_STRING(s))

namespace
{
const std::string &
SeriesDir()
{
  static const std::string dir = TOSTRING(DICOM_SERIES_INPUT);
  return dir;
}

unsigned int
CopyDicomSlices(const std::string & srcDir, const std::string & dstDir)
{
  itksys::SystemTools::MakeDirectory(dstDir);
  itksys::Directory directory;
  directory.Load(srcDir.c_str());
  unsigned int copied = 0;
  for (unsigned int i = 0; i < directory.GetNumberOfFiles(); ++i)
  {
    const std::string name = directory.GetFile(i);
    if (itksys::SystemTools::GetFilenameLastExtension(name) != ".dcm")
    {
      continue;
    }
    if (itksys::SystemTools::CopyFileAlways(srcDir + '/' + name, dstDir + '/' + name))
    {
      ++copied;
    }
  }
  return copied;
}

std::string
FirstDicomSlice(const std::string & srcDir)
{
  itksys::Directory directory;
  directory.Load(srcDir.c_str());
  for (unsigned int i = 0; i < directory.GetNumberOfFiles(); ++i)
  {
    const std::string name = directory.GetFile(i);
    if (itksys::SystemTools::GetFilenameLastExtension(name) == ".dcm")
    {
      return srcDir + '/' + name;
    }
  }
  return {};
}

// Copy one slice, overwriting Instance Number (0020,0013) so duplicate-IPP
// fixtures exercise the legacy ordering fallbacks.
bool
WriteWithInstanceNumber(const std::string & src, const std::string & dst, const char * instanceNumber)
{
  gdcm::Reader reader;
  reader.SetFileName(src.c_str());
  if (!reader.Read())
  {
    return false;
  }
  gdcm::DataElement de(gdcm::Tag(0x0020, 0x0013));
  de.SetVR(gdcm::VR::IS);
  de.SetByteValue(instanceNumber, static_cast<uint32_t>(std::strlen(instanceNumber)));
  reader.GetFile().GetDataSet().Replace(de);
  gdcm::Writer writer;
  writer.SetFileName(dst.c_str());
  writer.SetFile(reader.GetFile());
  return writer.Write();
}
} // namespace

TEST(GDCMSeriesFileNamesContract, SingleSeriesEnumerationAndGrouping)
{
  auto sf = itk::GDCMSeriesFileNames::New();
  sf->SetInputDirectory(SeriesDir());

  const std::vector<std::string> uids = sf->GetSeriesUIDs();
  ASSERT_EQ(uids.size(), 1u) << "DicomSeries is a single series";

  const std::vector<std::string> byUid = sf->GetFileNames(uids.front());
  EXPECT_GT(byUid.size(), 1u) << "Series has multiple slices";

  const std::vector<std::string> input = sf->GetInputFileNames();
  EXPECT_EQ(input, byUid) << "GetInputFileNames must return the (single) series, ordered identically";
}

TEST(GDCMSeriesFileNamesContract, OrderingReconstructsValidVolume)
{
  using ImageType = itk::Image<short, 3>;
  auto sf = itk::GDCMSeriesFileNames::New();
  sf->SetInputDirectory(SeriesDir());
  const std::vector<std::string> files = sf->GetInputFileNames();
  ASSERT_GT(files.size(), 1u);

  auto reader = itk::ImageSeriesReader<ImageType>::New();
  reader->SetImageIO(itk::GDCMImageIO::New());
  reader->SetFileNames(files);
  ASSERT_NO_THROW(reader->Update());

  const ImageType::Pointer image = reader->GetOutput();
  EXPECT_EQ(image->GetLargestPossibleRegion().GetSize()[2], files.size()) << "Each ordered slice becomes one z-plane";
  for (unsigned int d = 0; d < 3; ++d)
  {
    const double spacing = image->GetSpacing()[d];
    EXPECT_TRUE(std::isfinite(spacing) && spacing > 0.0) << "Spacing[" << d << "] must be positive and finite";
  }
}

TEST(GDCMSeriesFileNamesContract, RecursiveFlagHonored)
{
  const std::string root = TOSTRING(ITK_TEST_OUTPUT_DIR) + "/gdcmcontract_recursive";
  itksys::SystemTools::RemoveADirectory(root);
  itksys::SystemTools::MakeDirectory(root);
  ASSERT_GT(CopyDicomSlices(SeriesDir(), root + "/nested"), 1u);

  // Recursive is read lazily during BuildSeriesMap(); order relative to
  // SetInputDirectory does not matter.
  auto flat = itk::GDCMSeriesFileNames::New();
  flat->SetRecursive(false);
  flat->SetInputDirectory(root);
  EXPECT_TRUE(flat->GetInputFileNames().empty()) << "No DICOM directly under root; flat scan finds none";

  auto recursive = itk::GDCMSeriesFileNames::New();
  recursive->SetRecursive(true);
  recursive->SetInputDirectory(root);
  EXPECT_FALSE(recursive->GetInputFileNames().empty()) << "Recursive scan must descend into the subdirectory";
}

TEST(GDCMSeriesFileNamesContract, AmbiguousOrderingThrowsByDefault)
{
  const std::string root = TOSTRING(ITK_TEST_OUTPUT_DIR) + "/gdcmcontract_ambiguous_throw";
  itksys::SystemTools::RemoveADirectory(root);
  itksys::SystemTools::MakeDirectory(root);
  const std::string slice = FirstDicomSlice(SeriesDir());
  ASSERT_FALSE(slice.empty());
  // Two copies of the same slice: duplicate ImagePositionPatient.
  ASSERT_TRUE(itksys::SystemTools::CopyFileAlways(slice, root + "/copy_a.dcm"));
  ASSERT_TRUE(itksys::SystemTools::CopyFileAlways(slice, root + "/copy_b.dcm"));

  auto sf = itk::GDCMSeriesFileNames::New();
  EXPECT_TRUE(sf->GetFailOnAmbiguousOrdering());
  sf->SetInputDirectory(root);
  EXPECT_NO_THROW(sf->GetSeriesUIDs()) << "Enumeration does not order and must not throw";
  EXPECT_THROW(sf->GetInputFileNames(), itk::ExceptionObject);
}

TEST(GDCMSeriesFileNamesContract, LegacyFallbackUsesInstanceNumberThenFilename)
{
  const std::string slice = FirstDicomSlice(SeriesDir());
  ASSERT_FALSE(slice.empty());

  // Duplicate IPP with unique Instance Numbers: fallback must order by
  // Instance Number, chosen here to be the reverse of lexicographic order.
  const std::string byInstance = TOSTRING(ITK_TEST_OUTPUT_DIR) + "/gdcmcontract_fallback_instance";
  itksys::SystemTools::RemoveADirectory(byInstance);
  itksys::SystemTools::MakeDirectory(byInstance);
  ASSERT_TRUE(WriteWithInstanceNumber(slice, byInstance + "/z_first.dcm", "1 "));
  ASSERT_TRUE(WriteWithInstanceNumber(slice, byInstance + "/a_second.dcm", "2 "));

  auto sf = itk::GDCMSeriesFileNames::New();
  sf->FailOnAmbiguousOrderingOff();
  sf->SetInputDirectory(byInstance);
  const std::vector<std::string> instanceOrdered = sf->GetInputFileNames();
  ASSERT_EQ(instanceOrdered.size(), 2u);
  EXPECT_NE(instanceOrdered[0].find("z_first"), std::string::npos);
  EXPECT_NE(instanceOrdered[1].find("a_second"), std::string::npos);

  // Duplicate IPP and duplicate Instance Numbers: final fallback is
  // lexicographic filename order.
  const std::string byName = TOSTRING(ITK_TEST_OUTPUT_DIR) + "/gdcmcontract_fallback_filename";
  itksys::SystemTools::RemoveADirectory(byName);
  itksys::SystemTools::MakeDirectory(byName);
  ASSERT_TRUE(itksys::SystemTools::CopyFileAlways(slice, byName + "/copy_b.dcm"));
  ASSERT_TRUE(itksys::SystemTools::CopyFileAlways(slice, byName + "/copy_a.dcm"));

  auto lex = itk::GDCMSeriesFileNames::New();
  lex->FailOnAmbiguousOrderingOff();
  lex->SetInputDirectory(byName);
  const std::vector<std::string> nameOrdered = lex->GetInputFileNames();
  ASSERT_EQ(nameOrdered.size(), 2u);
  EXPECT_LT(nameOrdered[0], nameOrdered[1]);
}
